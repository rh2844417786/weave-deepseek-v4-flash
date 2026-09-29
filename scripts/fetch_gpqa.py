#!/usr/bin/env python3
"""Fetch the author's pinned archive and reproduce the local GPQA CSV files."""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import zipfile

REVISION = "56686c06f5e19865c153de0fdb11be3890014df7"
URL = f"https://raw.githubusercontent.com/idavidrein/gpqa/{REVISION}/dataset.zip"
ARCHIVE_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
# This is the public archive password documented by the dataset's author.
PASSWORD = b"deserted-untie-orchid"
CSV_SHA256 = {
    "gpqa_main.csv": "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3",
    "gpqa_diamond.csv": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
    "gpqa_extended.csv": "0926ee24949d02ed6748eb75a2611546c34479e30ddc42efd01d6f1681aaa48a",
}


def check(data, expected, name):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {name}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/gpqa")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        archive = args.data_root / "source/dataset.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            part = archive.with_suffix(".zip.part")
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "2", "--connect-timeout", "20", "--max-time", "120",
                "--output", str(part), URL,
            ], check=True)
            check(part.read_bytes(), ARCHIVE_SHA256, part.name)
            part.replace(archive)
        payload = archive.read_bytes()
        check(payload, ARCHIVE_SHA256, archive.name)
        raw = args.data_root / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            members = {Path(member.filename).name: member for member in zipped.infolist()}
            for name, expected in CSV_SHA256.items():
                data = zipped.read(members[name], pwd=PASSWORD)
                check(data, expected, name)
                destination = raw / name
                if not destination.exists():
                    destination.write_bytes(data)
                else:
                    check(destination.read_bytes(), expected, name)
    for name, expected in CSV_SHA256.items():
        path = args.data_root / "raw" / name
        check(path.read_bytes(), expected, name)
        print(f"verified {name} {expected}")


if __name__ == "__main__":
    main()
