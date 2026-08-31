"""Local inference against the fine-tuned checkpoint.

    python -m src.inference path/to/leaf.jpg

The classifier is a class because the model and processor are expensive to
load and need to stay resident - the API loads one instance at startup and
reuses it for every request.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src import config
from src.gpu_check import select_device

log = logging.getLogger(__name__)


class ModelNotFoundError(RuntimeError):
    """Raised when the checkpoint directory isn't there.

    Its own type so the API can turn it into a clear 503 rather than a
    generic 500 that tells the caller nothing.
    """


class BeanClassifier:
    def __init__(self, model_dir: Path | str = config.MODEL_DIR, device=None):
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise ModelNotFoundError(
                f"No model at {model_dir}. Run: python -m src.train"
            )

        self.model_dir = model_dir
        self.device = device or select_device()
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        metadata_path = model_dir / "training_metadata.json"
        self.metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )

        log.info("Loaded %s on %s", model_dir, self.device)

    @property
    def labels(self) -> list[str]:
        return [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        # Convert to RGB first: PNGs with alpha and greyscale JPEGs both arrive
        # with the wrong channel count and the processor won't fix that for us.
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probabilities = logits.softmax(dim=-1)[0]
        best = int(probabilities.argmax())

        return {
            "predicted_class": self.model.config.id2label[best],
            "confidence": round(float(probabilities[best]), 4),
            "probabilities": {
                label: round(float(probabilities[i]), 4)
                for i, label in enumerate(self.labels)
            },
            "device": str(self.device),
            "using_gpu": self.device.type == "cuda",
            "model_dir": str(self.model_dir),
            "base_model_revision": self.metadata.get("base_model_revision"),
            "dataset_revision": self.metadata.get("dataset_revision"),
        }

    def predict_file(self, path: Path | str) -> dict:
        with Image.open(path) as image:
            return self.predict(image)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Classify a bean leaf image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", default=config.MODEL_DIR)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU even when a GPU is present, for testing the fallback",
    )
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else None
    classifier = BeanClassifier(args.model_dir, device=device)
    print(json.dumps(classifier.predict_file(args.image), indent=2))


if __name__ == "__main__":
    main()
