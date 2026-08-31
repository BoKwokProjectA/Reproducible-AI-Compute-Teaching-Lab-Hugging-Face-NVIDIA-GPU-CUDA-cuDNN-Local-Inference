"""Beans dataset loading and preprocessing.

The dataset is AI-Lab-Makerere/beans: leaf photographs in three classes
(angular_leaf_spot, bean_rust, healthy), already split into train/validation/test
by the publisher, so we don't do our own splitting.
"""

from __future__ import annotations

import argparse

from datasets import load_dataset
from huggingface_hub import dataset_info, model_info

from src import config


def load_beans(split: str | None = None):
    """Load the dataset at the pinned revision.

    Passing revision=None asks for whatever is currently on main. That's fine
    while developing but it means a rerun months later may not get the same
    data, so config.DATASET_REVISION should be filled in before the results
    in the README are treated as reproducible.
    """
    return load_dataset(
        config.DATASET_ID,
        split=split,
        revision=config.DATASET_REVISION,
    )


def class_names(dataset) -> list[str]:
    """Class names in the order the model's output logits will use."""
    return dataset.features[config.LABEL_COLUMN].names


def make_collate_fn(processor):
    """Build the batch collator for a given image processor.

    The processor handles resizing to the size the ViT checkpoint expects and
    normalising with that checkpoint's own mean/std - hardcoding 224 and
    ImageNet statistics here would break the moment we swap the base model.
    """

    def collate(examples):
        import torch

        images = [example[config.IMAGE_COLUMN].convert("RGB") for example in examples]
        batch = processor(images=images, return_tensors="pt")
        batch["labels"] = torch.tensor(
            [example[config.LABEL_COLUMN] for example in examples], dtype=torch.long
        )
        return batch

    return collate


def show_revisions() -> None:
    """Print the current commit SHA of the dataset and base model repos.

    Copy these into src/config.py to pin them. Doing it this way means the
    pinned values are real hashes read from the Hub, not something anyone
    typed from memory.
    """
    dataset_sha = dataset_info(config.DATASET_ID).sha
    model_sha = model_info(config.BASE_MODEL_ID).sha

    print(f"DATASET_REVISION = {dataset_sha!r}      # {config.DATASET_ID}")
    print(f"BASE_MODEL_REVISION = {model_sha!r}     # {config.BASE_MODEL_ID}")
    print("\nPaste these into src/config.py, then commit.")


def show_summary() -> None:
    dataset = load_beans()
    labels = dataset["train"].features[config.LABEL_COLUMN]

    print(f"Dataset: {config.DATASET_ID}")
    print(f"Revision requested: {config.DATASET_REVISION or 'main (unpinned)'}")
    print(f"Classes ({labels.num_classes}): {', '.join(labels.names)}")
    for split_name in dataset:
        print(f"  {split_name:<12} {dataset[split_name].num_rows} images")

    example = dataset["train"][0]
    print(f"\nFirst training image: {example[config.IMAGE_COLUMN].size} px, "
          f"label={labels.int2str(example[config.LABEL_COLUMN])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the beans dataset.")
    parser.add_argument(
        "--show-revisions",
        action="store_true",
        help="print current Hub commit SHAs for pinning",
    )
    parser.add_argument(
        "--summary", action="store_true", help="print split sizes and classes"
    )
    args = parser.parse_args()

    if args.show_revisions:
        show_revisions()
    if args.summary or not args.show_revisions:
        show_summary()


if __name__ == "__main__":
    main()
