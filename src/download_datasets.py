"""
Standalone Utility Script: Download raw scRNA-seq supplementary files from NCBI GEO.
Kept separate from main.py to ensure ML pipeline reproducibility independent of network availability.

Usage:
    python download_datasets.py --output_dir ./data
"""
import argparse
import os
import requests
from tqdm import tqdm


GEO_DATASETS = {
    "GSE120575_Sade_Feldman_melanoma_TPM.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/"
        "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
    ),
    "GSE120575_patient_ID_single_cells.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/"
        "GSE120575_patient_ID_single_cells.txt.gz"
    ),
    "GSE115978_counts.csv.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/"
        "GSE115978_counts.csv.gz"
    ),
    "GSE115978_cell.annotations.csv.gz": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/"
        "GSE115978_cell.annotations.csv.gz"
    ),
}


def download_file(url: str, output_path: str, timeout: int = 60) -> None:
    """Stream-download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                bar.update(f.write(chunk))


def main(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading GEO melanoma scRNA-seq cohorts into: {output_dir}\n")

    for filename, url in GEO_DATASETS.items():
        target = os.path.join(output_dir, filename)
        if os.path.exists(target):
            print(f"[SKIP] {filename} already exists.")
            continue
        try:
            download_file(url, target)
            print(f"[OK]   Saved {target}")
        except Exception as exc:
            print(f"[FAIL] Could not download {filename}: {exc}")
            print(f"       Manual download URL: {url}\n")

    print("\nNote: For TCGA-SKCM clinical and RNA-seq survival data, download via the GDC Data Portal")
    print("(https://portal.gdc.cancer.gov/) and place the formatted table at ./data/tcga_skcm_survival.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GEO melanoma datasets.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Target directory for downloads.")
    args = parser.parse_args()
    main(args.output_dir)
