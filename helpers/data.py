"""
Resolve the configured dataset to concrete files and loaders.

`train.py` and `evaluate.py` differ only in which splits and jet counts they
ask for, so the per-dataset knowledge -- where a class's files live, and which
reader decodes them -- lives here instead of being branched at both call sites.
Both readers take the same keyword arguments and yield the same
``(batch, num_particles, 3)`` blocks, so `build_loader` only has to pick one.

Date: 2026-09-21
"""

import glob
import os
from typing import Iterable, List

from omegaconf import DictConfig
from torch.utils.data import DataLoader

import case_reader as cr
import jetsgame_reader as jr
from helpers.config import SUPPORTED_DATASETS

#: Dataset -> name used in "missing files" errors.
_DATASET_LABEL = {'jetclass': 'JetClass', 'jetsgame': 'JetsGame'}


def dataset_name(cfg: DictConfig) -> str:
    """Return the configured dataset, defaulting to JetClass for configs predating the key."""
    name = cfg.get('dataset', 'jetclass')
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset='{name}' is not supported. Use one of {SUPPORTED_DATASETS}.")
    return name


def split_label(cfg: DictConfig, split: str) -> str:
    """Directory the configured dataset reads `split` ('train', 'val' or 'test') from."""
    if dataset_name(cfg) == 'jetsgame':
        return jr.SPLITS[split][0]
    return f'flat_{split}' if cfg.flat else split


def class_files(cfg: DictConfig, split: str, sample: str) -> List[str]:
    """List the files holding one class for one split; empty when the class is absent."""
    if dataset_name(cfg) == 'jetsgame':
        return jr.jetsgame_files(
            cfg.data_dir, split, sample,
            cfg.get('jetsgame_pt_bin', '500GeV'), cfg.get('jetsgame_level', 'truth'),
        )
    return sorted(glob.glob(os.path.join(cfg.data_dir, split_label(cfg, split), sample, '*.h5')))


def require_files(cfg: DictConfig, splits: Iterable[str]) -> None:
    """
    Fail before any loading starts if a requested split is missing either class.

    Checked for every split up front so a missing validation or test set is
    reported immediately rather than after the training set has been read.
    """
    label = _DATASET_LABEL[dataset_name(cfg)]
    for split in splits:
        signal = class_files(cfg, split, cfg.signal)
        background = class_files(cfg, split, cfg.background)
        if not (signal and background):
            raise FileNotFoundError(
                f"Missing {label} files under {cfg.data_dir} for signal='{cfg.signal}', "
                f"background='{cfg.background}' (split '{split_label(cfg, split)}')"
            )


def build_loader(cfg: DictConfig, split: str, n_signal: int, n_background: int,
                 num_particles: int, *, batch_size: int, train: bool,
                 logger=None, seed: int = 0) -> DataLoader:
    """
    Build a balanced signal/background loader for one split of the configured dataset.

    Args:
        cfg: resolved run configuration.
        split: 'train', 'val' or 'test'.
        n_signal, n_background: jets requested per class.
        num_particles: constituents kept per jet.
        batch_size: jets per yielded batch.
        train: passed through to the reader for call-site compatibility.
        logger: optional loguru-style logger.
        seed: RNG seed for the signal/background shuffle.

    Returns:
        DataLoader: yields (jets, labels) batches of shape (batch, num_particles, 3).
    """
    loader = jr.JetsGameLoader if dataset_name(cfg) == 'jetsgame' else cr.OneP1QDataLoader
    return loader(
        signal_filelist=class_files(cfg, split, cfg.signal),
        background_filelist=class_files(cfg, split, cfg.background),
        n_signal=n_signal, n_background=n_background,
        batch_size=batch_size,
        input_shape=(num_particles, 3),
        train=train,
        normalize_pt=cfg.norm_pt,
        logger=logger,
        seed=seed,
    )
