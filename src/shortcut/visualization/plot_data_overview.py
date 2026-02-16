"""
Script to plot dataset overviews for MorphoMNIST, CheXpert, and Kermani OCT.

Example usage from terminal:

# Training data overview
python src/shortcut/visualization/plot_data_overview.py \
    --mnist_dir "/path/to/morpho_mnist/" \
    --chexpert_dir "/path/to/chexpert/" \
    --oct_dir "/path/to/kermani_oct/" \
    --output_fig "/path/to/save/train_data_distributions.pdf"

# Test data overview
python src/shortcut/visualization/plot_data_overview.py \
    --mode test \
    --mnist_dir "/path/to/morpho_mnist/" \
    --chexpert_dir "/path/to/chexpert/" \
    --oct_dir "/path/to/kermani_oct/" \
    --output_fig "/path/to/save/test_data_distributions.pdf"
"""

import argparse

from shortcut.data.chexpert import CheXpert
from shortcut.data.kermani_oct import (ConfoundedOCTDataset, get_confounder_fn,
                                       oct_normalize_transform)
from shortcut.data.morpho_mnist import MorphoMNISTCorrelated
from shortcut.utils import plot_data_overview, setup_fonts_and_style


def load_datasets(mode, mnist_dir, chexpert_dir, oct_dir):
    """
    Load datasets for either 'train' or 'test' mode.
    """
    if mode == "train":
        mnist_original = MorphoMNISTCorrelated(
            dataset_dir=mnist_dir, train=True, correlation_strength=None
        )
        mnist_train = MorphoMNISTCorrelated(
            dataset_dir=mnist_dir, train=True, correlation_strength=[95, 5]
        )
        mnist_groups = (mnist_original, mnist_train)

        chexpert_original = CheXpert(
            dataset_dir=chexpert_dir,
            split="train",
            attribute_labels=["Pleural Effusion", "Sex"],
            frontal=True,
            image_size=320,
            bucket_labels=None,
            bucket_samples=None,
        )
        chexpert_train = CheXpert(
            dataset_dir=chexpert_dir,
            split="train",
            attribute_labels=["Pleural Effusion", "Sex"],
            frontal=True,
            image_size=320,
            bucket_labels=[[1, 1], [0, 1], [1, 0], [0, 0]],
            bucket_samples=[8414, 450, 450, 8414],
        )
        chexpert_groups = (chexpert_original, chexpert_train)

        kermani_oct_original = ConfoundedOCTDataset(
            root_dir=oct_dir,
            split="train",
            classes_to_use=["normal", "drusen"],
            confounder_fn=get_confounder_fn("Radial notch"),
            correlation_strength={"drusen": 0.95, "normal": 0.05},
            transform=oct_normalize_transform,
            seed=42,
            bucket_labels=None,
            bucket_samples=None,
        )
        kermani_oct_train = ConfoundedOCTDataset(
            root_dir=oct_dir,
            split="train",
            classes_to_use=["normal", "drusen"],
            confounder_fn=get_confounder_fn("Radial notch"),
            correlation_strength={"drusen": 0.95, "normal": 0.05},
            transform=oct_normalize_transform,
            seed=42,
            bucket_labels=[[0, 0], [0, 1], [1, 0], [1, 1]],
            bucket_samples=[8185, 431, 431, 8185],
        )
        kermani_oct_groups = (kermani_oct_original, kermani_oct_train)

        dataset_groups = [mnist_groups, chexpert_groups, kermani_oct_groups]

    elif mode == "test":
        mnist_original = MorphoMNISTCorrelated(
            dataset_dir=mnist_dir, train=False, correlation_strength=None
        )
        mnist_inverse = MorphoMNISTCorrelated(
            dataset_dir=mnist_dir, train=False, correlation_strength=[5, 95]
        )
        mnist_balanced = MorphoMNISTCorrelated(
            dataset_dir=mnist_dir, train=False, correlation_strength=[50, 50]
        )
        mnist_groups = (mnist_original, mnist_balanced, mnist_inverse)

        chexpert_original = CheXpert(
            dataset_dir=chexpert_dir,
            split="test",
            attribute_labels=["Pleural Effusion", "Sex"],
            frontal=True,
            image_size=320,
            bucket_labels=None,
            bucket_samples=None,
        )
        chexpert_inverse = CheXpert(
            dataset_dir=chexpert_dir,
            split="test",
            attribute_labels=["Pleural Effusion", "Sex"],
            frontal=True,
            image_size=320,
            bucket_labels=[[1, 1], [0, 1], [1, 0], [0, 0]],
            bucket_samples=[155, 2924, 2924, 155],
        )
        chexpert_balanced = CheXpert(
            dataset_dir=chexpert_dir,
            split="test",
            attribute_labels=["Pleural Effusion", "Sex"],
            frontal=True,
            image_size=320,
            bucket_labels=[[1, 1], [0, 1], [1, 0], [0, 0]],
            bucket_samples=[2227, 2227, 2227, 2227],
        )
        chexpert_groups = (chexpert_original, chexpert_balanced, chexpert_inverse)

        kermani_oct_original = ConfoundedOCTDataset(
            root_dir=oct_dir,
            split="test",
            classes_to_use=["normal", "drusen"],
            confounder_fn=None,
            correlation_strength=None,
            transform=oct_normalize_transform,
            seed=42,
            bucket_labels=None,
            bucket_samples=None,
        )
        kermani_oct_inverse = ConfoundedOCTDataset(
            root_dir=oct_dir,
            split="test",
            classes_to_use=["normal", "drusen"],
            confounder_fn=get_confounder_fn("Radial notch"),
            correlation_strength={"drusen": 0.05, "normal": 0.95},
            transform=oct_normalize_transform,
            seed=42,
            bucket_labels=None,
            bucket_samples=None,
        )
        kermani_oct_balanced = ConfoundedOCTDataset(
            root_dir=oct_dir,
            split="test",
            classes_to_use=["normal", "drusen"],
            confounder_fn=get_confounder_fn("Radial notch"),
            correlation_strength={"drusen": 0.5, "normal": 0.5},
            transform=oct_normalize_transform,
            seed=42,
            bucket_labels=None,
            bucket_samples=None,
        )
        kermani_oct_groups = (
            kermani_oct_original,
            kermani_oct_balanced,
            kermani_oct_inverse,
        )

        dataset_groups = [mnist_groups, chexpert_groups, kermani_oct_groups]

    else:
        raise ValueError("Mode must be 'train' or 'test'")

    return dataset_groups


def main():
    parser = argparse.ArgumentParser(
        description="Plot dataset overviews for MorphoMNIST, CheXpert, and Kermani OCT."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: train or test dataset overview",
    )
    parser.add_argument(
        "--mnist_dir", type=str, required=True, help="Path to MorphoMNIST dataset"
    )
    parser.add_argument(
        "--chexpert_dir", type=str, required=True, help="Path to CheXpert dataset"
    )
    parser.add_argument(
        "--oct_dir", type=str, required=True, help="Path to Kermani OCT dataset"
    )
    parser.add_argument(
        "--output_fig",
        type=str,
        required=True,
        help="Path to save the output figure (PDF)",
    )
    args = parser.parse_args()

    setup_fonts_and_style(style_path=None)

    # Load dataset groups
    dataset_groups = load_datasets(
        args.mode, args.mnist_dir, args.chexpert_dir, args.oct_dir
    )

    # Define levels
    levels = [
        {
            "y1": [0, 1],
            "y2": [0, 1],
            "y1_names": ["0-4", "5-9"],
            "y2_names": ["thin", "thick"],
        },
        {
            "y1": [0, 1],
            "y2": [0, 1],
            "y1_names": ["healthy", "pleural \n effusion"],
            "y2_names": ["female", "male"],
        },
        {
            "y1": [0, 1],
            "y2": [0, 1],
            "y1_names": ["healthy", "drusen"],
            "y2_names": ["no filter", "filter"],
        },
    ]

    # Create figure
    fig = plot_data_overview(
        dataset_groups=dataset_groups,
        dataset_names=["MorphoMNIST", "CheXpert", "OCT"],
        levels=levels,
        cmaps=["gray", "gray", "gray"],
        examples_per_cell=1 if args.mode == "train" else None,
        figsize_per_dataset=(1.8, 1.8),
        hspace=0.3,
        wspace=0.05 if args.mode == "test" else 0.1,
    )

    # Save figure
    fig.savefig(
        args.output_fig, bbox_inches="tight", dpi=300, facecolor="white", pad_inches=0.0
    )
    print(f"Figure saved successfully to {args.output_fig}")


if __name__ == "__main__":
    main()
