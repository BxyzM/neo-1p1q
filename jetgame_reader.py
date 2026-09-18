"""
Load and batch the JetGame 1P1Q feature cache for binary jet-tagging tasks.

JetGame ships as one HDF5 file with `train`/`valid`/`test` groups, unlike the
per-class JetClass directories that `case_reader.py` reads, so class selection
here is by label rather than by which directory a file came from. That is what
makes `w_vs_qcd` expressible: the cache carries QCD, W and Top in every split.

Two differences from `case_reader.py` are load-bearing and silent if missed:

1. Column order. This repository's circuits index particle features through
   `helpers.utils.particleFeatureNames == ['eta', 'phi', 'pt']`, but the cache
   stores `['pt_fraction', 'delta_eta', 'delta_phi']`. The columns are permuted
   on read; without that, pt would be fed into the eta rotation with no error.
2. Scaling. `case_reader.fixed_rescale` maps raw JetClass units (pt up to
   ~3 TeV, eta/phi in +-0.8) onto the circuit's expected ranges. JetGame values
   are already a pt fraction and eta/phi differences, so no rescaling is
   applied here -- re-applying JetClass's would distort them.

Author: Aritra Bal (ETP), JetGame reader added on the jetgame-exp003 branch
"""

import h5py
from pennylane import numpy as np
import numpy as nnp
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Tuple, Union

import helpers.utils as ut

# Raw JetGame label encoding, asserted against the file on read.
QCD, W, TOP = 0, 1, 2
# task -> (signal label, background label); signal is always trained as class 1.
TASKS = {
    'top_vs_qcd': (TOP, QCD),
    'w_vs_qcd': (W, QCD),
}
CACHE_COLUMNS = ['pt_fraction', 'delta_eta', 'delta_phi']


def _column_permutation() -> List[int]:
    """Indices that reorder cache columns into this repo's (eta, phi, pt)."""
    mapping = {'eta': 'delta_eta', 'phi': 'delta_phi', 'pt': 'pt_fraction'}
    return [CACHE_COLUMNS.index(mapping[name]) for name in ut.particleFeatureNames]


class JetGameDataset(IterableDataset):
    """
    Balanced binary dataset built from one JetGame 1P1Q cache.

    Reads `n_signal` jets of the task's signal class (label 1) and
    `n_background` of its background class (label 0) from one split, keeps the
    leading `n_qubits` particles, concatenates and shuffles.

    Args:
        cache (str): path to the JetGame .h5 feature cache.
        split (str): 'train', 'valid', or 'test'.
        task (str): key of TASKS, e.g. 'top_vs_qcd' or 'w_vs_qcd'.
        n_signal (int): number of signal jets to read.
        n_background (int): number of background jets to read.
        batch_size (int): jets per yielded batch.
        input_shape (tuple[int]): (particles per jet to keep, features per particle).
        logger: optional loguru-style logger; falls back to print.
        seed (int): RNG seed for class subsampling and the shuffle.

    Yields:
        Tuple[np.ndarray, np.ndarray]: a batch of jets and their integer labels.
    """

    def __init__(self, cache: str, split: str, task: str,
                 n_signal: int, n_background: int, batch_size: int = 32,
                 input_shape: tuple = (10, 3), logger=None, seed: int = 0):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Available: {sorted(TASKS)}")
        if split not in ('train', 'valid', 'test'):
            raise ValueError("split must be train, valid, or test")
        self.cache = cache
        self.split = split
        self.task = task
        self.n_signal = int(n_signal)
        self.n_background = int(n_background)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_qubits = input_shape[0]
        self.logger = logger
        self.seed = seed
        self._data, self._labels = self._materialise()

    def _log(self, msg: str) -> None:
        """Route a message through the logger if present, else stdout."""
        self.logger.info(msg) if self.logger is not None else print(msg)

    def _read_class(self, handle, raw_labels, label_value: int, n_target: int,
                    assigned: int, permutation) -> np.ndarray:
        """Read up to `n_target` jets of one raw class, tagged with `assigned`."""
        rows = nnp.flatnonzero(raw_labels == label_value)
        if not len(rows):
            raise ValueError(
                f"{self.cache}:{self.split} contains no jets with raw label {label_value}"
            )
        if len(rows) < n_target:
            self._log(
                f"WARNING: requested {n_target} jets of raw label {label_value} "
                f"but only {len(rows)} are available"
            )
            n_target = len(rows)
        generator = nnp.random.default_rng(self.seed + label_value)
        rows = nnp.sort(generator.choice(rows, size=n_target, replace=False))
        # h5py needs an increasing selection; sorting above keeps the read
        # sequential rather than a slow scattered point selection.
        features = handle[self.split]['particle_features'][rows, :self.n_qubits, :]
        features = nnp.asarray(features)[:, :, permutation]
        return features, nnp.full(len(features), assigned, dtype=int)

    def _materialise(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read both classes, concatenate, shuffle, and log the resulting split."""
        signal_label, background_label = TASKS[self.task]
        permutation = _column_permutation()
        with h5py.File(self.cache, 'r') as handle:
            if self.split not in handle:
                raise ValueError(f"{self.cache} has no '{self.split}' split")
            columns = handle.attrs.get('particle_feature_columns')
            if columns is not None:
                import json
                stored = json.loads(columns) if isinstance(columns, str) else list(columns)
                if list(stored) != CACHE_COLUMNS:
                    raise ValueError(
                        f"Cache column order {stored} differs from the expected "
                        f"{CACHE_COLUMNS}; the permutation in this reader would be wrong"
                    )
            raw_labels = nnp.asarray(handle[self.split]['labels'][:])
            if not nnp.isin(raw_labels, [QCD, W, TOP]).all():
                raise ValueError("Expected canonical raw labels: 0=QCD, 1=W, 2=Top")
            available = handle[self.split]['particle_features'].shape[1]
            if self.n_qubits > available:
                raise ValueError(
                    f"Requested {self.n_qubits} particles; cache stores {available}"
                )
            self._log(
                f"Reading '{self.task}' from {self.split}/: signal=raw label "
                f"{signal_label}, background=raw label {background_label}; "
                f"keeping the {self.n_qubits} leading particles each"
            )
            sig_data, sig_labels = self._read_class(
                handle, raw_labels, signal_label, self.n_signal, 1, permutation)
            bg_data, bg_labels = self._read_class(
                handle, raw_labels, background_label, self.n_background, 0, permutation)

        data = nnp.concatenate([sig_data, bg_data], axis=0)
        labels = nnp.concatenate([sig_labels, bg_labels], axis=0)
        idx = nnp.random.default_rng(self.seed).permutation(len(data))
        data, labels = data[idx], labels[idx]

        total = len(data)
        self._log(
            f"JetGame loader finished: {total} jets read -- {len(sig_data)} signal "
            f"(label 1), {len(bg_data)} background (label 0)"
        )
        if total:
            eta, phi, pt = (ut.getIndex('particle', name) for name in ('eta', 'phi', 'pt'))
            self._log(
                f"  cache ranges (no rescaling applied): "
                f"pt [{float(data[..., pt].min()):.3f}, {float(data[..., pt].max()):.3f}], "
                f"eta [{float(data[..., eta].min()):.3f}, {float(data[..., eta].max()):.3f}], "
                f"phi [{float(data[..., phi].min()):.3f}, {float(data[..., phi].max()):.3f}]"
            )
        return data, labels

    def __iter__(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        '''Yield the pre-loaded, pre-shuffled jets in batches of `batch_size`.'''
        for i in range(0, len(self._data), self.batch_size):
            batch_data = torch.from_numpy(nnp.asarray(self._data[i:i + self.batch_size])).float()
            batch_labels = self._labels[i:i + self.batch_size]
            yield np.array(batch_data, requires_grad=False), np.array(batch_labels, dtype=int, requires_grad=False)

    def __len__(self) -> int:
        """Return the number of batches yielded, including a partial final batch."""
        return (len(self._data) + self.batch_size - 1) // self.batch_size


def JetGameDataLoader(input_shape: tuple = (10, 3), **kwargs) -> DataLoader:
    '''
    Build a DataLoader over a balanced JetGameDataset.

    Mirrors `case_reader.OneP1QDataLoader` so `train.py` can swap readers
    without changing how it consumes batches. Pass `cache`, `split`, `task`,
    `n_signal`, `n_background` (plus optional `batch_size`, `logger`, `seed`).

    Returns:
        DataLoader: yields (jets, labels) batches; batching is done by the dataset.
    '''
    print(f"Will read only {input_shape[0]} particles per jet")
    dset = JetGameDataset(input_shape=input_shape, **kwargs)
    return DataLoader(dset, batch_size=None)  # batching is managed by the dataset
