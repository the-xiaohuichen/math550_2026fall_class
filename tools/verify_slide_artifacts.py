"""Verify an editable teaching deck and its classroom PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    deck = parser.add_mutually_exclusive_group(required=True)
    deck.add_argument("--key", type=Path)
    deck.add_argument("--pptx", type=Path)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, required=True)
    parser.add_argument("--expected-equations", type=int)
    return parser.parse_args()


def verify_keynote(path: Path, expected_equations: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a native zipped Keynote package: {path}")
    with zipfile.ZipFile(path) as package:
        corrupt_member = package.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt Keynote member: {corrupt_member}")
        if expected_equations is not None:
            equation_count = sum(
                bool(re.fullmatch(r"Data/equation-\d+\.pdf", name))
                for name in package.namelist()
            )
            if equation_count != expected_equations:
                raise ValueError(
                    f"Expected {expected_equations} native Keynote equations, "
                    f"found {equation_count}: {path}"
                )


def verify_powerpoint(path: Path, expected_equations: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a zipped PowerPoint package: {path}")
    with zipfile.ZipFile(path) as package:
        corrupt_member = package.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt PowerPoint member: {corrupt_member}")
        if "ppt/presentation.xml" not in package.namelist():
            raise ValueError(f"Missing ppt/presentation.xml: {path}")
        if expected_equations is not None:
            equation_count = sum(
                package.read(name).count(b"<a14:m ")
                for name in package.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            )
            if equation_count != expected_equations:
                raise ValueError(
                    f"Expected {expected_equations} editable OfficeMath objects, "
                    f"found {equation_count}: {path}"
                )


def verify_pdf(path: Path, expected_pages: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError(f"Invalid PDF structure: {path}")
    page_count = len(re.findall(rb"/Type\s*/Page\b", data))
    if page_count != expected_pages:
        raise ValueError(
            f"Expected {expected_pages} PDF pages, found {page_count}: {path}"
        )


def main() -> None:
    args = parse_args()
    if args.key is not None:
        verify_keynote(args.key, args.expected_equations)
        deck_label = "native Keynote package"
    else:
        verify_powerpoint(args.pptx, args.expected_equations)
        deck_label = "editable PowerPoint package"
    verify_pdf(args.pdf, args.expected_pages)
    equation_label = (
        f" with {args.expected_equations} native equations"
        if args.expected_equations is not None
        else ""
    )
    print(
        f"Verified {deck_label}{equation_label} and "
        f"{args.expected_pages}-page PDF."
    )


if __name__ == "__main__":
    main()
