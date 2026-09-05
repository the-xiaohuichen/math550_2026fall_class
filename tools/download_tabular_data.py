"""Download and verify the UCI datasets used by the tabular module."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "tabular" / "uci"
DATA_DIR = PROJECT_ROOT / "data" / "tabular"

DATASETS: dict[str, dict[str, object]] = {
    "wdbc": {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "source_page": (
            "https://archive.ics.uci.edu/dataset/17/"
            "breast-cancer-wisconsin-diagnostic"
        ),
        "doi": "10.24432/C5DW2B",
        "license": "CC BY 4.0",
        "citation": (
            "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). "
            "Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine "
            "Learning Repository."
        ),
        "archive": {
            "name": "breast-cancer-wisconsin-diagnostic.zip",
            "url": (
                "https://archive.ics.uci.edu/static/public/17/"
                "breast+cancer+wisconsin+diagnostic.zip"
            ),
            "sha256": (
                "bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af"
            ),
            "bytes": 51284,
        },
        "files": {
            "wdbc.data": {
                "sha256": (
                    "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a"
                ),
                "bytes": 124103,
            },
            "wdbc.names": {
                "sha256": (
                    "840e04e3f20f8a5b326892f3b9cbc01c4cd6f7e6c597630b701ef6c0ac79f5ef"
                ),
                "bytes": 4708,
            },
        },
        "validation": {
            "rows": 569,
            "fields_per_row": 32,
            "target_column": 1,
            "class_counts": {"B": 357, "M": 212},
        },
    },
    "wine": {
        "dataset": "Wine",
        "source_page": "https://archive.ics.uci.edu/dataset/109/wine",
        "doi": "10.24432/C5PC7J",
        "license": "CC BY 4.0",
        "citation": (
            "Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine "
            "Learning Repository."
        ),
        "archive": {
            "name": "wine.zip",
            "url": "https://archive.ics.uci.edu/static/public/109/wine.zip",
            "sha256": (
                "2bae62c4481220623579d4c4fb36b55652b6b75e06e49fa1981b8198362dfdab"
            ),
            "bytes": 6038,
        },
        "files": {
            "Index": {
                "sha256": (
                    "c24d1f17df97bdde234913bb0a3334227215eefd0ad3d6a9988151d49971cba7"
                ),
                "bytes": 105,
            },
            "wine.data": {
                "sha256": (
                    "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659"
                ),
                "bytes": 10782,
            },
            "wine.names": {
                "sha256": (
                    "f1b84f2ef845e0bdebf13e14fa7a213e56de4f1baa40c5974dbd1ee51c5ae710"
                ),
                "bytes": 3036,
            },
        },
        "validation": {
            "rows": 178,
            "fields_per_row": 14,
            "target_column": 0,
            "class_counts": {"1": 59, "2": 71, "3": 48},
        },
    },
    "dry_bean": {
        "dataset": "Dry Bean",
        "source_page": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        "doi": "10.24432/C50S4B",
        "license": "CC BY 4.0",
        "citation": (
            "Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C50S4B."
        ),
        "archive": {
            "name": "dry-bean-dataset.zip",
            "url": (
                "https://archive.ics.uci.edu/static/public/602/"
                "dry+bean+dataset.zip"
            ),
            "sha256": (
                "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
            ),
            "bytes": 4738776,
        },
        "files": {
            "Dry_Bean_Dataset.arff": {
                "archive_path": "DryBeanDataset/Dry_Bean_Dataset.arff",
                "sha256": (
                    "b2a4a76a2aedfb8ed415adfc1bfc70b5f202cb00cb72e500766e364c14834014"
                ),
                "bytes": 3813205,
            },
            "Dry_Bean_Dataset.txt": {
                "archive_path": "DryBeanDataset/Dry_Bean_Dataset.txt",
                "sha256": (
                    "20413356adbeda9da589738cceb9387787970d8cfb41049f3b48c960c05b274d"
                ),
                "bytes": 3339,
            },
        },
        "validation": {
            "format": "arff",
            "data_file": "Dry_Bean_Dataset.arff",
            "rows": 13611,
            "fields_per_row": 17,
            "target_column": 16,
            "class_counts": {
                "BARBUNYA": 1322,
                "BOMBAY": 522,
                "CALI": 1630,
                "DERMASON": 3546,
                "HOROZ": 1928,
                "SEKER": 2027,
                "SIRA": 2636,
            },
        },
    },
}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path) -> None:
    """Download an archive atomically with an explicit course user agent."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "MATH550-course-data/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def verify_digest(path: Path, expected: str) -> None:
    """Fail loudly when a source or extracted file is not the pinned snapshot."""

    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, observed {observed}"
        )


def extract_dataset(key: str, specification: dict[str, object]) -> None:
    """Fetch one pinned archive and extract only its documented files."""

    archive = specification["archive"]
    assert isinstance(archive, dict)
    archive_path = CACHE_DIR / str(archive["name"])
    expected_archive_digest = str(archive["sha256"])
    if not archive_path.is_file() or sha256(archive_path) != expected_archive_digest:
        download_archive(str(archive["url"]), archive_path)
    verify_digest(archive_path, expected_archive_digest)

    file_specs = specification["files"]
    assert isinstance(file_specs, dict)
    output_dir = DATA_DIR / key
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as source:
        available = set(source.namelist())
        required_sources = {
            str(raw_file_specification.get("archive_path", name))
            for name, raw_file_specification in file_specs.items()
            if isinstance(raw_file_specification, dict)
        }
        missing = required_sources - available
        if missing:
            raise RuntimeError(f"Archive {archive_path} is missing {sorted(missing)}")
        for name, raw_file_specification in file_specs.items():
            assert isinstance(raw_file_specification, dict)
            output_path = output_dir / name
            source_name = str(raw_file_specification.get("archive_path", name))
            output_path.write_bytes(source.read(source_name))
            verify_digest(output_path, str(raw_file_specification["sha256"]))


def verify_rows(
    key: str,
    rows: list[list[str]],
    validation: dict[str, object],
) -> None:
    """Check row width, finite features, and class counts."""

    expected_rows = int(validation["rows"])
    expected_fields = int(validation["fields_per_row"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"{key} has {len(rows)} rows; expected {expected_rows}")
    if any(len(row) != expected_fields for row in rows):
        raise RuntimeError(f"{key} contains a row with an unexpected field count")
    target_column = int(validation["target_column"])
    if not all(
        math.isfinite(float(value))
        for row in rows
        for column, value in enumerate(row)
        if column != target_column
    ):
        raise RuntimeError(f"{key} contains a non-finite feature")
    observed_counts = dict(Counter(row[target_column] for row in rows))
    if observed_counts != validation["class_counts"]:
        raise RuntimeError(
            f"{key} class counts are {observed_counts}; "
            f"expected {validation['class_counts']}"
        )


def verify_csv_dataset(key: str, specification: dict[str, object]) -> None:
    """Read and validate one header-free comma-separated source file."""

    validation = specification["validation"]
    assert isinstance(validation, dict)
    data_filename = "wdbc.data" if key == "wdbc" else "wine.data"
    with (DATA_DIR / key / data_filename).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))
    verify_rows(key, rows, validation)


def verify_arff_dataset(key: str, specification: dict[str, object]) -> None:
    """Read and validate the comma-separated data section of an ARFF file."""

    validation = specification["validation"]
    assert isinstance(validation, dict)
    path = DATA_DIR / key / str(validation["data_file"])
    rows: list[list[str]] = []
    in_data = False
    with path.open(encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not in_data:
                in_data = line.lower() == "@data"
                continue
            if line and not line.startswith("%"):
                rows.append(next(csv.reader([line])))
    if not in_data:
        raise RuntimeError(f"{key} ARFF source does not contain an @DATA marker")
    verify_rows(key, rows, validation)


def verify_dataset(key: str, specification: dict[str, object]) -> None:
    """Dispatch validation according to the preserved source format."""

    validation = specification["validation"]
    assert isinstance(validation, dict)
    if validation.get("format") == "arff":
        verify_arff_dataset(key, specification)
    else:
        verify_csv_dataset(key, specification)


def write_manifest() -> Path:
    """Write deterministic provenance for the downloaded source snapshots."""

    manifest = {
        "schema_version": 1,
        "verified_on": "2026-09-03",
        "datasets": {},
    }
    for key, specification in DATASETS.items():
        record = dict(specification)
        record["local_directory"] = f"data/tabular/{key}"
        manifest["datasets"][key] = record
    output_path = DATA_DIR / "download_manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def run() -> Path:
    """Download, extract, validate, and record the course datasets."""

    for key, specification in DATASETS.items():
        extract_dataset(key, specification)
        verify_dataset(key, specification)
    return write_manifest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest_path = run()
    print(
        "Verified WDBC (569 x 30), Wine (178 x 13), and "
        f"Dry Bean (13,611 x 16): {manifest_path}"
    )


if __name__ == "__main__":
    main()
