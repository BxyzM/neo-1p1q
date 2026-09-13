"""
Generate tiny synthetic JetClass-shaped HDF5 fixtures for the container's
baked-in smoke tests (tests.test_end_to_end.TestBalancedJetClassLoader).
Matches the real data contract: jetConstituentsList (N, 100, 3) with
(eta, phi, pt) per particle, particles sorted by descending pt per jet;
jetFeatures (N, 1) with jet pt in column 0. Values are synthetic, not
physically meaningful -- only shape/ordering/dtype matter for this test.

Author: Aritra Bal (ETP)
Date: 2026-09-13
"""
import argparse
import os

import h5py
import numpy as np


def make_class_file(path: str, n_jets: int, n_particles: int, seed: int) -> None:
    """Write one HDF5 file with n_jets synthetic jets to path."""
    rng = np.random.default_rng(seed)
    pt = np.sort(rng.exponential(scale=20.0, size=(n_jets, n_particles)), axis=1)[:, ::-1]
    eta = rng.uniform(-0.8, 0.8, size=(n_jets, n_particles))
    phi = rng.uniform(-0.8, 0.8, size=(n_jets, n_particles))
    constituents = np.stack([eta, phi, pt], axis=-1).astype(np.float64)
    jet_features = pt.sum(axis=1, keepdims=True).astype(np.float64)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("jetConstituentsList", data=constituents)
        f.create_dataset("jetFeatures", data=jet_features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="JetClass-style root directory to write into")
    parser.add_argument("--n-jets", type=int, default=100, help="Jets per class")
    parser.add_argument("--n-particles", type=int, default=100, help="Particles per jet")
    args = parser.parse_args()

    classes = {"TTBar_": 0, "ZJetsToNuNu": 1}
    for sample, seed in classes.items():
        out_path = os.path.join(args.out_dir, "test", sample, f"{sample}_000.h5")
        make_class_file(out_path, args.n_jets, args.n_particles, seed=seed)
        print(f"wrote {args.n_jets} jets -> {out_path}")


if __name__ == "__main__":
    main()
