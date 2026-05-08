#!/usr/bin/env python3
"""Small helper to download large model files outside of git.

Usage:
    python scripts/download_model.py --url <URL> --dest Notebook/banmani_model/pytorch_model.bin

The script streams the download to avoid loading large files into memory.
"""
import argparse
import os
import sys

try:
    import requests
except Exception:
    print("requests is required. Install with: pip install requests")
    sys.exit(1)


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = resp.headers.get("content-length")
    if total is not None:
        total = int(total)

    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                print(f"\rDownloading... {percent}%", end="", flush=True)
    if total:
        print("\rDownload complete.       ")
    else:
        print("Download complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="URL to download the model from")
    p.add_argument("--dest", default="Notebook/banmani_model/pytorch_model.bin", help="Destination path to save the model")
    args = p.parse_args()
    download(args.url, args.dest)


if __name__ == "__main__":
    main()
