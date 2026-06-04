"""Download the RAVDESS speech audio subset.

RAVDESS is hosted on Zenodo (record 1188976). The audio-only speech zip
is ~250 MB. We download it, unpack into data/raw/, and verify by parsing
filenames.

Usage:
    python scripts/download_ravdess.py

Idempotent: if data/raw already contains valid clips, the script exits
early without re-downloading.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Make src/ importable when run as a script from project root.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from voicemood.config import RAW_DIR, ensure_dirs  # noqa: E402
from voicemood.dataset import discover_clips  # noqa: E402

# The "Audio_Speech_Actors_01-24.zip" file from Zenodo record 1188976.
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_NAME = "ravdess_speech.zip"
EXPECTED_CLIP_COUNT = 1440   # 24 actors x 60 utterances


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with dest.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def _unzip(zip_path: Path, target: Path) -> None:
    print(f"Unzipping into {target}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)


def main() -> int:
    ensure_dirs()

    existing = discover_clips(RAW_DIR)
    if len(existing) >= EXPECTED_CLIP_COUNT:
        print(f"Found {len(existing)} clips in {RAW_DIR}; skipping download.")
        return 0

    zip_path = RAW_DIR / ZIP_NAME
    if not zip_path.exists():
        try:
            _download(RAVDESS_URL, zip_path)
        except requests.RequestException as exc:
            print(f"ERROR: download failed: {exc}", file=sys.stderr)
            print(
                "If your network blocks Zenodo, you can manually download\n"
                f"  {RAVDESS_URL}\n"
                f"and place the file at {zip_path}, then re-run this script.",
                file=sys.stderr,
            )
            return 1

    _unzip(zip_path, RAW_DIR)

    clips = discover_clips(RAW_DIR)
    print(f"Discovered {len(clips)} valid clips.")
    if len(clips) < EXPECTED_CLIP_COUNT:
        print(
            f"WARNING: expected {EXPECTED_CLIP_COUNT} clips but found {len(clips)}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
