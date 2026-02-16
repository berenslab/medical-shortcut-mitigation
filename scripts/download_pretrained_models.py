import argparse
import zipfile
from pathlib import Path

import requests

ZENODO_ID = "18630074"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_ID}/files"

FOLDER_TO_ZIP = {
    "morpho_mnist": "morpho_mnist.zip",
    "chexpert": "chexpert.zip",
    "oct": "oct.zip",
    "prevalence_ablation": "prevalence_ablation.zip",
}


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url} → {dest} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")


def extract_zip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted {zip_path} → {out_dir}")
    zip_path.unlink()


def main(args):
    out_dir = Path(args.out).resolve()

    for folder in args.folders:
        if folder not in FOLDER_TO_ZIP:
            print(f"Warning: unknown folder '{folder}'")
            continue
        zip_name = FOLDER_TO_ZIP[folder]
        url = f"{ZENODO_BASE_URL}/{zip_name}"
        zip_path = out_dir / zip_name

        download(url, zip_path)
        extract_zip(zip_path, out_dir)

    print(f"Done. Folders are available in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download selected dataset/model/result folders from Zenodo."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out",
        help="Target output directory (default: ./out)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        choices=["morpho_mnist", "chexpert", "oct", "prevalence_ablation"],
        default=["morpho_mnist"],
        help="Which folders to download (default: morpho_mnist)",
    )
    args = parser.parse_args()
    main(args)
