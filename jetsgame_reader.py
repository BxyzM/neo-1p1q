"""
Load and batch the JetsGame jet samples (Top / WW / QCD) for the 1P1Q classifier.

JetsGame ships one gzipped JSON-lines file per class, pt bin and simulation level.
Every line is one jet, encoded as a list of four-momenta ``{"E","px","py","pz"}``
in GeV: element 0 is the jet itself and the remaining elements are its
constituents, so summing the constituents reproduces element 0 exactly.

That differs from JetClass in three ways this module has to absorb, because the
circuit expects the same ``(N, n_qubits, 3)`` block of jet-relative
``(eta, phi, pt)`` from either dataset:

  * constituents come as ``(E, px, py, pz)``, not ``(eta, phi, pt)``;
  * they arrive in no particular order, so the hardest ones must be selected
    rather than sliced off the front;
  * they carry absolute detector angles, so the jet axis has to be subtracted
    and delta-phi wrapped into ``[-pi, pi)``.

Everything downstream of that -- the pt/jet_pt or fixed_rescale scaling, class
balancing, deterministic shuffling and internal batching -- is inherited from
`case_reader.CASEJetClassDataset` unchanged.

Date: 2026-09-21
"""

import gzip
import json
import os
from typing import Iterator, List

import numpy as nnp
from pennylane import numpy as np
from torch.utils.data import DataLoader

import helpers.utils as ut
from case_reader import CASEJetClassDataset

#: Sample names, as they appear in the JetsGame filenames.
SAMPLES = ('QCD', 'WW', 'Top')
#: Jet pt thresholds each sample is generated at.
PT_BINS = ('200GeV', '500GeV', '2TeV')
#: Simulation level -> filename suffix. 'truth' is the full simulation (no suffix).
LEVELS = {'truth': '', 'hadron': '_hadron', 'parton': '_parton'}
#: Pipeline split -> (JetsGame directory, filename prefix).
SPLITS = {'train': ('train', ''), 'val': ('valid', 'valid_'), 'test': ('test', 'test_')}

_GZIP_MAGIC = b'\x1f\x8b'
_LFS_POINTER = b'version https://git-lfs.github.com/spec/v1'


def jetsgame_path(data_dir: str, split: str, sample: str,
                 pt_bin: str = '500GeV', level: str = 'truth') -> str:
    """
    Build the path of one JetsGame file.

    Args:
        data_dir: root holding the ``train/``, ``valid/`` and ``test/`` directories.
        split: pipeline split name, one of ``SPLITS`` ('train', 'val' or 'test').
        sample: class name, one of ``SAMPLES``.
        pt_bin: jet pt threshold, one of ``PT_BINS``.
        level: simulation level, one of ``LEVELS``.

    Returns:
        str: ``<data_dir>/<dir>/<prefix><sample>_<pt_bin><suffix>.json.gz``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got '{split}'")
    if sample not in SAMPLES:
        raise ValueError(f"JetsGame sample must be one of {list(SAMPLES)}, got '{sample}'")
    if pt_bin not in PT_BINS:
        raise ValueError(f"JetsGame pt bin must be one of {list(PT_BINS)}, got '{pt_bin}'")
    if level not in LEVELS:
        raise ValueError(f"JetsGame level must be one of {sorted(LEVELS)}, got '{level}'")
    directory, prefix = SPLITS[split]
    return os.path.join(data_dir, directory, f"{prefix}{sample}_{pt_bin}{LEVELS[level]}.json.gz")


def jetsgame_files(data_dir: str, split: str, sample: str,
                  pt_bin: str = '500GeV', level: str = 'truth') -> List[str]:
    """Return the single-element file list for one class, or ``[]`` if it is absent."""
    path = jetsgame_path(data_dir, split, sample, pt_bin, level)
    return [path] if os.path.isfile(path) else []


def _iter_jets(file_path: str, max_rows: int) -> Iterator[list]:
    """
    Yield up to `max_rows` parsed jets from a gzipped JSON-lines JetsGame file.

    The distribution is a git-lfs repository, so an un-pulled file is a short
    text pointer rather than gzip data. That is detected here and reported with
    the fix, instead of surfacing as an opaque decompression error.
    """
    with open(file_path, 'rb') as raw:
        head = raw.read(len(_LFS_POINTER))
    if not head.startswith(_GZIP_MAGIC):
        if head.startswith(_LFS_POINTER):
            raise ValueError(
                f"{file_path} is a git-lfs pointer, not jet data; "
                f"run 'git lfs pull' in {os.path.dirname(os.path.dirname(file_path))}"
            )
        raise ValueError(f"{file_path} is not gzip data; expected a JetsGame .json.gz file")
    with gzip.open(file_path, 'rt') as stream:
        for row, line in enumerate(stream):
            if row >= max_rows:
                return
            yield json.loads(line)


class JetsGameDataset(CASEJetClassDataset):
    """
    Balanced binary dataset built from JetsGame ``.json.gz`` files.

    Same construction and output contract as `CASEJetClassDataset` -- pass the
    signal and background file lists, get ``(N, n_qubits, 3)`` jet-relative
    ``(eta, phi, pt)`` batches -- with only the per-file decoding replaced.
    """

    LOADER_NAME = 'JetsGame'
    #: JetsGame clusters anti-kt R=1.0 jets, so its eta/phi window is wider than JetClass's.
    ASSUMED_LIMITS = ut.jetsgame_assumed_limits

    def _class_name(self, filelist: List[str]) -> str:
        """JetsGame keeps one file per class, so the filename names the class."""
        return os.path.basename(filelist[0]).replace('.json.gz', '') if filelist else '?'

    def _jet_to_particles(self, record: list) -> nnp.ndarray:
        """
        Turn one JetsGame jet record into ``(n_qubits, 3)`` of jet-relative (eta, phi, pt).

        Jets with fewer than `n_qubits` constituents are zero-padded at the end,
        matching how JetClass pads its fixed-width constituent block.
        """
        four_vectors = nnp.array(
            [[p['E'], p['px'], p['py'], p['pz']] for p in record], dtype=nnp.float64
        )
        jet, constituents = four_vectors[0], four_vectors[1:]
        # Element 0 is the jet, not a constituent. Verify rather than assume: mistaking
        # the two silently shifts every angle by the jet axis and loses the hardest particle.
        if not nnp.allclose(constituents.sum(axis=0), jet, rtol=1.0e-3, atol=1.0e-3 * abs(jet[0])):
            raise ValueError(
                "JetsGame record does not start with its jet four-momentum: the "
                "constituents do not sum to element 0"
            )

        px, py, pz = constituents[:, 1], constituents[:, 2], constituents[:, 3]
        pt = nnp.hypot(px, py)
        # Unlike JetClass, JetsGame constituents are unordered; take the hardest
        # n_qubits with a stable sort so the same file always yields the same jets.
        keep = nnp.argsort(-pt, kind='stable')[:self.n_qubits]
        pt = pt[keep]
        eta = nnp.arcsinh(pz[keep] / nnp.clip(pt, 1.0e-12, None))
        phi = nnp.arctan2(py[keep], px[keep])

        jet_pt = nnp.hypot(jet[1], jet[2])
        jet_eta = nnp.arcsinh(jet[3] / jet_pt)
        jet_phi = nnp.arctan2(jet[2], jet[1])

        particles = nnp.zeros((self.n_qubits, 3), dtype=nnp.float64)
        kept = len(pt)
        particles[:kept, self.eta_index] = eta - jet_eta
        particles[:kept, self.phi_index] = (phi - jet_phi + nnp.pi) % (2 * nnp.pi) - nnp.pi
        particles[:kept, self.pt_index] = pt / jet_pt if self.normalize_pt else pt
        return particles

    def _load_file(self, file_path: str, max_rows: int) -> np.ndarray:
        """
        Load one JetsGame ``.json.gz`` file and return its preprocessed jets.

        Keeps the `n_qubits` hardest constituents per jet, then applies exactly
        the scaling `CASEJetClassDataset._load_file` applies: pt divided by jet
        pt when `normalize_pt`, otherwise the fixed rescaling, and fixed
        rescaling for eta and phi either way.

        Args:
            file_path: gzipped JSON-lines file to read.
            max_rows: maximum number of leading jets to read.

        Returns:
            np.ndarray: shape (M, n_qubits, 3).
        """
        jets = [self._jet_to_particles(record) for record in _iter_jets(file_path, max_rows)]
        if not jets:
            return np.empty((0, self.n_qubits, 3))
        jet_etaphipt = nnp.stack(jets)
        padded = int(nnp.sum(jet_etaphipt[..., self.pt_index] == 0.0))
        if padded:
            self._log(
                f"{os.path.basename(file_path)}: zero-padded {padded} particle slots across "
                f"{len(jets)} jets (fewer than {self.n_qubits} constituents available)"
            )

        if not self.normalize_pt:
            jet_etaphipt[..., self.pt_index] = self.fixed_rescale(
                jet_etaphipt[..., self.pt_index], epsilon=self.epsilon, type='pt')
        jet_etaphipt[..., self.eta_index] = self.fixed_rescale(
            jet_etaphipt[..., self.eta_index], epsilon=self.epsilon, type='eta')
        jet_etaphipt[..., self.phi_index] = self.fixed_rescale(
            jet_etaphipt[..., self.phi_index], epsilon=self.epsilon, type='phi')
        return np.array(jet_etaphipt)


def JetsGameLoader(input_shape: tuple[int] = (10, 3), train: bool = True, **kwargs) -> DataLoader:
    """
    Build a DataLoader over a balanced signal/background `JetsGameDataset`.

    Mirrors `case_reader.OneP1QDataLoader`: pass `signal_filelist`,
    `background_filelist`, `n_signal`, `n_background` (plus optional
    `batch_size`, `normalize_pt`, `logger`, `seed`) via kwargs.

    Returns:
        DataLoader: yields (jets, labels) batches; batching is done by the dataset.
    """
    print(f"Will read only {input_shape[0]} particles per jet")
    dset = JetsGameDataset(input_shape=input_shape, train=train, **kwargs)
    return DataLoader(dset, batch_size=None)  # batching is managed by the dataset
