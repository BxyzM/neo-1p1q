"""Bounded HDF5 reads of the existing JetClass/JetGame 1P1Q caches.

No preprocessing is recomputed: cache identity, canonical row indices and
event identifiers accompany the selected angles throughout the experiment.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import fcntl
import tempfile

import h5py
import numpy as np


@dataclass
class FeatureSplit:
    angles: np.ndarray
    labels: np.ndarray
    rows: np.ndarray
    fingerprints: np.ndarray

    def metadata(self):
        digest = hashlib.sha256()
        for array in (self.rows, self.labels, self.angles, self.fingerprints):
            digest.update(np.ascontiguousarray(array).view(np.uint8))
        return {
            "events": len(self.labels),
            "background": int(np.count_nonzero(self.labels == 0)),
            "signal": int(np.count_nonzero(self.labels == 1)),
            "selected_data_sha256": digest.hexdigest(),
        }


def read_rows(dataset, rows, block_rows=65536, particles=None):
    """Read unique rows in bounded contiguous blocks, preserving request order.

    Random h5py point selections scale poorly. Read only blocks intersecting
    the selection, never dataset[:] for large selections. For a sparse block,
    bound its read to the first and last requested row. Peak scratch storage
    is at most block_rows * the selected row width, independent of cache size.
    """
    rows = np.asarray(rows)
    if rows.size and not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("rows must contain integer indices")
    rows = rows.astype(np.int64, copy=False)
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError("rows must be a one-dimensional array of unique indices")
    if len(rows) and (rows.min() < 0 or rows.max() >= dataset.shape[0]):
        raise ValueError("Selected row is outside the dataset")
    chunks = getattr(dataset, "chunks", None)
    if chunks and chunks[0] <= block_rows:
        # Align to physical row chunks to avoid decompressing the same chunk
        # twice at adjacent block boundaries (JetClass chunks are 35157 rows).
        block_rows = max(1, block_rows // chunks[0]) * chunks[0]
    tail = dataset.shape[1:]
    if particles is not None:
        if dataset.ndim != 3 or not 1 <= particles <= dataset.shape[1]:
            raise ValueError("Requested particle count is unavailable in the cache")
        tail = (particles, dataset.shape[2])
    result = np.empty((len(rows), *tail), dtype=dataset.dtype)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    start = 0
    while start < len(rows):
        stop_row = (int(sorted_rows[start]) // block_rows + 1) * block_rows
        stop = int(np.searchsorted(sorted_rows, stop_row))
        first, last = int(sorted_rows[start]), int(sorted_rows[stop - 1]) + 1
        selection = slice(first, last) if particles is None else (slice(first, last), slice(0, particles), slice(None))
        block = np.asarray(dataset[selection])
        result[order[start:stop]] = block[sorted_rows[start:stop] - first]
        start = stop
    return result


def cache_metadata(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format_name") != "polarization.1p1q-features" or not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"Not a complete 1P1Q feature cache: {path}")
        source = json.loads(handle.attrs["configuration"])
        if source.get("phi_difference") != "wrapped":
            raise ValueError("The depth study requires a cache with wrapped delta-phi")
        return {
            "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "format": str(handle.attrs["format_name"]), "source_configuration": source,
        }


def load_feature_split(path, split, max_events, seed, particles, block_rows=65536):
    """Match polarization's Top/QCD subset RNG and label convention exactly."""
    if split not in ("train", "valid", "test"):
        raise ValueError("split must be train, valid, or test")
    if max_events is not None and (type(max_events) is not int or max_events < 4 or max_events % 2):
        raise ValueError("max_events must be null or an even integer >= 4")
    cache_metadata(path)
    with h5py.File(path, "r") as handle:
        group = handle[split]
        raw = group["labels"][:]
        if raw.ndim != 1 or not np.isin(raw, [0, 1, 2]).all():
            raise ValueError("Expected canonical raw labels: 0=QCD, 1=W, 2=Top")
        background, signal = np.flatnonzero(raw == 0), np.flatnonzero(raw == 2)
        if not len(background) or not len(signal):
            raise ValueError(f"{split} must contain both QCD and Top jets")
        if max_events is not None:
            count = max_events // 2
            if min(len(background), len(signal)) < count:
                raise ValueError(f"{split}: requested {count} per class; only {len(background)} QCD and {len(signal)} Top available")
            offset = {"train": 0, "valid": 1, "test": 2}[split] + 10000
            generator = np.random.default_rng(seed + offset)
            background = generator.choice(background, size=count, replace=False)
            signal = generator.choice(signal, size=count, replace=False)
        elif len(background) != len(signal):
            raise ValueError(f"{split}: full split is unbalanced; specify an explicit balanced event budget")
        rows = np.concatenate((background, signal)).astype(np.int64)
        features = group["encoding_angles"]
        fingerprints = group["event_fingerprints"]
        if features.ndim != 3 or features.shape[0] != len(raw) or features.shape[2] != 2 or fingerprints.shape != (len(raw), 16):
            raise ValueError(f"{split}: malformed or misaligned feature/identity arrays")
        angles = read_rows(features, rows, block_rows, particles).astype(np.float32, copy=False)
        identities = read_rows(fingerprints, rows, block_rows)
    if not np.isfinite(angles).all():
        raise ValueError(f"{split}: nonfinite encoding angles")
    return FeatureSplit(
        angles, np.concatenate((np.zeros(len(background)), np.ones(len(signal)))).astype(np.uint8),
        rows, identities,
    )


def check_split_overlap(train, valid):
    """JetGame uses content fingerprints; JetClass stores split-specific IDs.

    The latter check detects duplicated IDs, but cannot establish absence of
    physical duplicate events. This limitation belongs in the study report.
    """
    a = np.ascontiguousarray(train.fingerprints).view("V16").ravel()
    b = np.ascontiguousarray(valid.fingerprints).view("V16").ravel()
    if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b):
        raise ValueError("Duplicate event identifiers within a selected split")
    if len(np.intersect1d(a, b)):
        raise ValueError("Training and validation contain overlapping event identifiers")


def prepare_splits(output_dir, dataset_config, seed, particles, block_rows):
    """Prepare each selected subset once, shared by readout/depth workers.

    Files are immutable after publication and read through NumPy memory maps.
    A lock plus directory rename lets independent launchers safely share them.
    """
    identity = {
        "cache": cache_metadata(dataset_config["cache"]),
        "selection": dataset_config, "seed": seed, "particles": particles,
        "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    parent = Path(output_dir) / "data"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / key
    with (parent / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            train = load_feature_split(dataset_config["cache"], "train", dataset_config["max_train_events"], seed, particles, block_rows)
            valid = load_feature_split(dataset_config["cache"], "valid", dataset_config["max_valid_events"], seed, particles, block_rows)
            check_split_overlap(train, valid)
            with tempfile.TemporaryDirectory(dir=parent, prefix=f".{key}-") as temporary:
                stage = Path(temporary) / "prepared"
                stage.mkdir()
                metadata = {"identity": identity, "splits": {}}
                for name, split in (("train", train), ("valid", valid)):
                    metadata["splits"][name] = split.metadata()
                    for field in ("angles", "labels", "rows", "fingerprints"):
                        np.save(stage / f"{name}_{field}.npy", getattr(split, field), allow_pickle=False)
                (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                stage.rename(target)
        metadata = json.loads((target / "metadata.json").read_text())
        if metadata["identity"] != identity:
            raise ValueError(f"Prepared subset identity mismatch: {target}")
    splits = {}
    for name in ("train", "valid"):
        split = FeatureSplit(**{
            field: np.load(target / f"{name}_{field}.npy", mmap_mode="r", allow_pickle=False)
            for field in ("angles", "labels", "rows", "fingerprints")
        })
        if split.metadata() != metadata["splits"][name]:
            raise ValueError(f"Prepared subset contents changed: {target}/{name}")
        splits[name] = split
    return splits, {**metadata, "prepared_directory": str(target.resolve())}
