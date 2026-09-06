"""Public image data and leakage-safe preprocessing for generative lessons."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import shutil
import struct
from urllib.request import urlopen

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    source_url: str
    doi: str
    license: str
    n_rows: int
    n_features: int
    image_shape: tuple[int, int]
    pixel_range: tuple[int, int]
    classes: int
    note: str


@dataclass
class LatentDigits:
    train_images: np.ndarray
    test_images: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_latent: np.ndarray
    test_latent: np.ndarray
    pca: PCA
    metadata: DatasetMetadata

    def inverse_transform(self, latent: np.ndarray) -> np.ndarray:
        flat = self.pca.inverse_transform(np.asarray(latent, dtype=float))
        return np.clip(flat.reshape(-1, *self.metadata.image_shape), 0.0, 1.0)


@dataclass(frozen=True)
class MNISTData:
    """Original 28-by-28 MNIST train/test arrays and provenance metadata."""

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    metadata: DatasetMetadata
    file_md5: dict[str, str]


MNIST_BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _md5(destination) == expected_md5:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual_md5 = _md5(temporary)
    if actual_md5 != expected_md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"checksum mismatch for {destination.name}: "
            f"expected {expected_md5}, got {actual_md5}"
        )
    temporary.replace(destination)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051 or rows != 28 or columns != 28:
            raise ValueError(f"invalid MNIST image header in {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * columns
    if values.size != expected:
        raise ValueError(f"truncated MNIST image file: {path}")
    return values.reshape(count, rows, columns).copy()


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"invalid MNIST label header in {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"truncated MNIST label file: {path}")
    return values.copy()


def load_mnist(cache_dir: Path | str = Path(".cache/mnist"), download: bool = True) -> MNISTData:
    """Load original MNIST with checksum verification and no torchvision dependency.

    The primary dataset description is Yann LeCun, Corinna Cortes, and Chris
    Burges's MNIST page. Files are fetched from the mirror used by torchvision;
    the pinned MD5 values match torchvision's public dataset implementation.
    Pixel arrays stay as ``uint8`` so callers choose their own model scaling.
    """

    cache_dir = Path(cache_dir)
    if download:
        for filename, checksum in MNIST_FILES.items():
            _download_verified(MNIST_BASE_URL + filename, cache_dir / filename, checksum)
    missing = [name for name in MNIST_FILES if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "MNIST files are missing; rerun with download=True: " + ", ".join(missing)
        )
    actual_md5 = {name: _md5(cache_dir / name) for name in MNIST_FILES}
    mismatched = [
        name for name, checksum in MNIST_FILES.items() if actual_md5[name] != checksum
    ]
    if mismatched:
        raise ValueError("MNIST checksum mismatch: " + ", ".join(mismatched))

    train_images = _read_idx_images(cache_dir / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels(cache_dir / "train-labels-idx1-ubyte.gz")
    test_images = _read_idx_images(cache_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx_labels(cache_dir / "t10k-labels-idx1-ubyte.gz")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("MNIST image and label counts do not agree")
    metadata = DatasetMetadata(
        name="MNIST handwritten digits",
        source_url="https://yann.lecun.org/exdb/mnist/",
        doi="not assigned on the official distribution page",
        license="no explicit license stated on the official distribution page",
        n_rows=len(train_images) + len(test_images),
        n_features=28 * 28,
        image_shape=(28, 28),
        pixel_range=(0, 255),
        classes=10,
        note=(
            "60,000 training and 10,000 test images; centered and size-normalized "
            "in the original MNIST release"
        ),
    )
    return MNISTData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        metadata=metadata,
        file_md5=actual_md5,
    )


def load_latent_digits(
    n_components: int = 8,
    test_fraction: float = 0.20,
    random_state: int = 550,
) -> LatentDigits:
    """Load scikit-learn digits and fit a whitened PCA on training rows only."""

    if not 2 <= n_components <= 64:
        raise ValueError("n_components must lie between 2 and 64")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    bunch = load_digits()
    images = bunch.images.astype(float) / 16.0
    flat = images.reshape(len(images), -1)
    indices = np.arange(len(images))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_fraction,
        stratify=bunch.target,
        random_state=random_state,
    )
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    train_latent = pca.fit_transform(flat[train_indices])
    test_latent = pca.transform(flat[test_indices])
    metadata = DatasetMetadata(
        name="scikit-learn digits (UCI Optical Recognition subset)",
        source_url="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        doi="10.24432/C50P49",
        license="CC BY 4.0",
        n_rows=len(images),
        n_features=flat.shape[1],
        image_shape=(8, 8),
        pixel_range=(0, 16),
        classes=10,
        note=(
            "scikit-learn distributes 1,797 8x8 digit images drawn from the "
            "test portion of the UCI collection"
        ),
    )
    return LatentDigits(
        train_images=images[train_indices],
        test_images=images[test_indices],
        train_labels=np.asarray(bunch.target)[train_indices],
        test_labels=np.asarray(bunch.target)[test_indices],
        train_latent=train_latent,
        test_latent=test_latent,
        pca=pca,
        metadata=metadata,
    )
