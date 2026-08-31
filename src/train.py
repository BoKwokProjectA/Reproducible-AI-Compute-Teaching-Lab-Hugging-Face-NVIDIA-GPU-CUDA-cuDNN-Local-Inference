"""Fine-tune a pretrained ViT on the beans dataset.

Deliberately a plain PyTorch loop rather than transformers.Trainer. Trainer
would be shorter, but it hides the device placement and the memory accounting,
and those are the parts of this project worth being able to point at.

    python -m src.train --epochs 3 --batch-size 16

Training is short by design - the goal is a working pipeline and a usable
checkpoint, not a competitive classifier.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src import config
from src.dataset import class_names, load_beans, make_collate_fn
from src.gpu_check import collect_gpu_info, select_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", force=True)
log = logging.getLogger("train")


def build_model(labels: list[str]):
    """Load the pretrained ViT and swap in a head sized for our classes.

    The in21k checkpoint has no usable classification head for us, so
    ignore_mismatched_sizes lets Transformers discard it and initialise a
    fresh 3-class one. The warning it prints is expected.
    """
    return AutoModelForImageClassification.from_pretrained(
        config.BASE_MODEL_ID,
        revision=config.BASE_MODEL_REVISION,
        num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id={name: i for i, name in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions = model(**batch).logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()
    return correct / total if total else 0.0


def train_one_epoch(model, loader, optimiser, device, scaler=None) -> float:
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss = model(**batch).loss
            loss.backward()
            optimiser.step()

        running_loss += loss.item()
        if step % 10 == 0:
            log.info("  step %d  loss %.4f", step, running_loss / step)

    return running_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ViT on beans.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="truncate the training split, useful for a quick smoke test",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="mixed precision; roughly halves VRAM use on a supported GPU",
    )
    parser.add_argument("--output-dir", default=str(config.MODEL_DIR))
    args = parser.parse_args()

    device = select_device()
    gpu_info = collect_gpu_info()
    log.info("Device: %s", device)
    if device.type == "cpu":
        log.warning("Running on CPU. This will be slow but will still produce a model.")

    if args.fp16 and device.type != "cuda":
        log.warning("--fp16 ignored: needs CUDA")
        args.fp16 = False

    dataset = load_beans()
    labels = class_names(dataset["train"])
    log.info("Classes: %s", ", ".join(labels))

    train_split = dataset["train"]
    if args.max_train_samples:
        train_split = train_split.select(range(args.max_train_samples))

    processor = AutoImageProcessor.from_pretrained(
        config.BASE_MODEL_ID, revision=config.BASE_MODEL_REVISION
    )
    collate = make_collate_fn(processor)

    # num_workers=2 rather than 0 because decoding JPEGs on one thread starves
    # the GPU; pin_memory only helps when there's a GPU to copy to.
    loader_kwargs = {
        "collate_fn": collate,
        "num_workers": 2,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_split, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        dataset["validation"], batch_size=args.batch_size, **loader_kwargs
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=args.batch_size, **loader_kwargs
    )

    model = build_model(labels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None

    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        log.info("Epoch %d/%d", epoch, args.epochs)
        loss = train_one_epoch(model, train_loader, optimiser, device, scaler)
        accuracy = evaluate(model, val_loader, device)
        log.info("  train loss %.4f  val accuracy %.4f", loss, accuracy)
        history.append({"epoch": epoch, "train_loss": loss, "val_accuracy": accuracy})

        if device.type == "cuda":
            log.info(
                "  VRAM allocated %.0f MiB, peak %.0f MiB",
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.max_memory_allocated() / 1024**2,
            )

    duration = time.time() - started
    test_accuracy = evaluate(model, test_loader, device)
    log.info("Test accuracy: %.4f (trained in %.0fs)", test_accuracy, duration)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Provenance record. Written from the run that just happened, so the
    # numbers in it are always the numbers that produced these weights.
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": config.BASE_MODEL_ID,
        "base_model_revision": config.BASE_MODEL_REVISION,
        "dataset": config.DATASET_ID,
        "dataset_revision": config.DATASET_REVISION,
        "classes": labels,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "fp16": args.fp16,
            "max_train_samples": args.max_train_samples,
        },
        "train_samples": train_split.num_rows,
        "history": history,
        "test_accuracy": test_accuracy,
        "training_seconds": round(duration, 1),
        "device": str(device),
        "gpu": gpu_info["devices"][0]["name"] if gpu_info["devices"] else None,
        "torch_version": gpu_info["torch_version"],
        "torch_cuda_build": gpu_info["torch_cuda_build"],
        "cudnn_version": gpu_info["cudnn_version"],
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    log.info("Saved model and metadata to %s", output_dir)


if __name__ == "__main__":
    main()
