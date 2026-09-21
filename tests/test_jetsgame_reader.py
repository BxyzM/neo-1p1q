"""
Verify JetsGame decoding, hardest-constituent selection, scaling, and dispatch.

The synthetic fixtures are built from known (pt, eta, phi) triples, so each test
can invert the fixed rescaling and assert the reader recovered exactly the
kinematics that went in -- rather than only checking shapes.

Date: 2026-09-21
"""

import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as onp
from omegaconf import OmegaConf

import helpers.data as data_helpers
import helpers.utils as ut
import jetsgame_reader as jr

# The distribution directory on disk is spelled 'JetGame'; the dataset is JetsGame.
_JETSGAME_DIR = os.environ.get('JETSGAME_DIR', '/ceph/mbinder/JetGame/data-v1.0.0')

_ETA = ut.getIndex('particle', 'eta')
_PHI = ut.getIndex('particle', 'phi')
_PT = ut.getIndex('particle', 'pt')


def massless(pt: float, eta: float, phi: float) -> dict:
    """One massless four-momentum record with exactly these (pt, eta, phi)."""
    px, py, pz = pt * onp.cos(phi), pt * onp.sin(phi), pt * onp.sinh(eta)
    return {'E': float(onp.sqrt(px * px + py * py + pz * pz)),
            'px': float(px), 'py': float(py), 'pz': float(pz)}


def write_jets(path: str, jets: list[list[dict]]) -> None:
    """Write JetsGame's format: one gzipped JSON line per jet, jet four-vector first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for constituents in jets:
            total = {key: sum(p[key] for p in constituents) for key in ('E', 'px', 'py', 'pz')}
            stream.write(json.dumps([total] + constituents) + '\n')


def unscale(scaled, kind: str, epsilon: float = 1.0e-4):
    """Invert JetsGameDataset.fixed_rescale, so a test can compare physical values."""
    low, high = jr.JetsGameDataset.ASSUMED_LIMITS[kind]
    minimum = ut.feature_limits[kind]['min']
    maximum = ut.feature_limits[kind]['max'] - epsilon
    return (scaled - minimum) / (maximum - minimum) * (high - low) + low


class _JetsGameFixture(unittest.TestCase):
    """Writes signal/background fixtures into a temporary JetsGame-shaped tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def path(self, split: str, sample: str) -> str:
        return jr.jetsgame_path(str(self.root), split, sample)

    def dataset(self, signal, background, **kwargs):
        """Build a JetsGameDataset over two written fixtures, with test-friendly defaults."""
        options = dict(n_signal=len(signal), n_background=len(background), batch_size=8,
                       input_shape=(3, 3), normalize_pt=True, seed=0)
        options.update(kwargs)
        write_jets(self.path('train', 'Top'), signal)
        write_jets(self.path('train', 'QCD'), background)
        return jr.JetsGameDataset(
            signal_filelist=[self.path('train', 'Top')],
            background_filelist=[self.path('train', 'QCD')],
            **options,
        )


class TestJetsGamePaths(unittest.TestCase):
    """jetsgame_path() must reproduce the distribution's filenames and reject typos."""

    def test_split_prefixes_and_level_suffixes(self) -> None:
        root = '/data'
        self.assertEqual(jr.jetsgame_path(root, 'train', 'Top', '500GeV', 'truth'),
                         '/data/train/Top_500GeV.json.gz')
        self.assertEqual(jr.jetsgame_path(root, 'val', 'QCD', '200GeV', 'hadron'),
                         '/data/valid/valid_QCD_200GeV_hadron.json.gz')
        self.assertEqual(jr.jetsgame_path(root, 'test', 'WW', '2TeV', 'parton'),
                         '/data/test/test_WW_2TeV_parton.json.gz')

    def test_unknown_values_are_rejected(self) -> None:
        for kwargs, message in (
            (dict(split='validation'), 'split must be one of'),
            (dict(sample='TTBar_'), 'JetsGame sample must be one of'),
            (dict(pt_bin='1TeV'), 'JetsGame pt bin must be one of'),
            (dict(level='detector'), 'JetsGame level must be one of'),
        ):
            call = dict(data_dir='/data', split='train', sample='Top',
                        pt_bin='500GeV', level='truth')
            call.update(kwargs)
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                jr.jetsgame_path(**call)

    def test_missing_file_yields_empty_list(self) -> None:
        self.assertEqual(jr.jetsgame_files('/nonexistent', 'train', 'Top'), [])


class TestJetsGameDecoding(_JetsGameFixture):
    """The (E, px, py, pz) -> jet-relative (eta, phi, pt) conversion must be exact."""

    def test_recovers_known_kinematics(self) -> None:
        constituents = [massless(300.0, 0.10, 0.20),
                        massless(100.0, -0.15, 0.35),
                        massless(50.0, 0.40, -0.10)]
        dataset = self.dataset([constituents], [constituents])
        jet = onp.asarray(dataset._data[0])

        four = onp.array([[p['px'], p['py'], p['pz']] for p in constituents])
        total = four.sum(axis=0)
        jet_pt = onp.hypot(total[0], total[1])
        jet_eta = onp.arcsinh(total[2] / jet_pt)
        jet_phi = onp.arctan2(total[1], total[0])

        for row, (pt, eta, phi) in enumerate([(300.0, 0.10, 0.20), (100.0, -0.15, 0.35),
                                              (50.0, 0.40, -0.10)]):
            self.assertAlmostEqual(unscale(jet[row, _ETA], 'eta'), eta - jet_eta, places=6)
            self.assertAlmostEqual(unscale(jet[row, _PHI], 'phi'), phi - jet_phi, places=6)
            self.assertAlmostEqual(jet[row, _PT], pt / jet_pt, places=6)

    def test_delta_phi_wraps_into_principal_range(self) -> None:
        """Constituents straddling the +/-pi seam keep a small delta-phi."""
        constituents = [massless(300.0, 0.0, onp.pi - 0.05),
                        massless(200.0, 0.0, -onp.pi + 0.05),
                        massless(100.0, 0.0, onp.pi - 0.15)]
        jet = onp.asarray(self.dataset([constituents], [constituents])._data[0])
        delta_phi = unscale(jet[:, _PHI], 'phi')
        self.assertTrue(onp.all(onp.abs(delta_phi) < 0.3),
                        f"delta-phi did not wrap: {delta_phi}")

    def test_keeps_hardest_constituents_in_descending_pt(self) -> None:
        """JetsGame constituents arrive unordered, so the reader must sort, not slice."""
        scrambled = [massless(20.0, 0.1, 0.1), massless(400.0, 0.2, -0.2),
                     massless(80.0, -0.3, 0.05), massless(5.0, 0.4, 0.4),
                     massless(150.0, -0.1, 0.25)]
        jet = onp.asarray(self.dataset([scrambled], [scrambled])._data[0])
        jet_pt = onp.hypot(*onp.array([[p['px'], p['py']] for p in scrambled]).sum(axis=0))
        onp.testing.assert_allclose(jet[:, _PT], onp.array([400.0, 150.0, 80.0]) / jet_pt, rtol=1e-6)

    def test_short_jets_are_zero_padded(self) -> None:
        """A jet with fewer constituents than qubits pads trailing slots with zeros."""
        short = [massless(300.0, 0.1, 0.2), massless(120.0, -0.2, 0.1)]
        jet = onp.asarray(self.dataset([short], [short], input_shape=(4, 3))._data[0])
        self.assertEqual(jet[2, _PT], 0.0)
        self.assertEqual(jet[3, _PT], 0.0)
        for row in (2, 3):
            self.assertAlmostEqual(unscale(jet[row, _ETA], 'eta'), 0.0, places=9)
            self.assertAlmostEqual(unscale(jet[row, _PHI], 'phi'), 0.0, places=9)

    def test_fixed_pt_rescaling_when_not_normalising(self) -> None:
        """With normalize_pt=False, pt goes through the shared fixed rescaling."""
        constituents = [massless(300.0, 0.1, 0.2), massless(100.0, -0.1, 0.1),
                        massless(50.0, 0.2, -0.2)]
        jet = onp.asarray(
            self.dataset([constituents], [constituents], normalize_pt=False)._data[0])
        onp.testing.assert_allclose(unscale(jet[:, _PT], 'pt'),
                                    onp.array([300.0, 100.0, 50.0]), rtol=1e-6)


class TestJetsGameFailures(_JetsGameFixture):
    """Malformed inputs must fail loudly rather than feed the circuit wrong particles."""

    def test_record_without_leading_jet_is_rejected(self) -> None:
        path = self.path('train', 'Top')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, 'wt') as stream:  # every element a constituent, no jet header
            stream.write(json.dumps([massless(300.0, 0.1, 0.2), massless(100.0, 0.0, 0.0)]) + '\n')
        with self.assertRaisesRegex(ValueError, 'does not start with its jet four-momentum'):
            jr.JetsGameDataset(signal_filelist=[path], background_filelist=[path],
                              n_signal=1, n_background=1, batch_size=1, input_shape=(2, 3))

    def test_git_lfs_pointer_is_reported(self) -> None:
        path = self.path('train', 'Top')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(
            'version https://git-lfs.github.com/spec/v1\n'
            'oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n'
            'size 134277952\n'
        )
        with self.assertRaisesRegex(ValueError, 'git-lfs pointer'):
            jr.JetsGameDataset(signal_filelist=[path], background_filelist=[path],
                              n_signal=1, n_background=1, batch_size=1, input_shape=(2, 3))

    def test_non_gzip_file_is_reported(self) -> None:
        path = self.path('train', 'Top')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text('[]\n')
        with self.assertRaisesRegex(ValueError, 'is not gzip data'):
            jr.JetsGameDataset(signal_filelist=[path], background_filelist=[path],
                              n_signal=1, n_background=1, batch_size=1, input_shape=(2, 3))


class TestJetsGameBalancing(_JetsGameFixture):
    """Class balancing, labelling and batching are inherited and must still hold."""

    def _jets(self, count: int, offset: float) -> list[list[dict]]:
        rng = onp.random.default_rng(int(offset))
        return [[massless(float(pt), float(eta), float(phi))
                 for pt, eta, phi in zip(rng.uniform(10, 400, 6),
                                         rng.uniform(-0.5, 0.5, 6),
                                         rng.uniform(-0.5, 0.5, 6))]
                for _ in range(count)]

    def test_equal_split_labels_and_batch_sizes(self) -> None:
        loader_dataset = self.dataset(self._jets(40, 1.0), self._jets(40, 2.0), batch_size=18)
        counts, batch_sizes, total = {0: 0, 1: 0}, [], 0
        first_shape = None
        for data, labels in loader_dataset:
            first_shape = first_shape or tuple(data.shape[1:])
            batch_sizes.append(data.shape[0])
            total += data.shape[0]
            for label in labels.tolist():
                counts[int(label)] += 1
        self.assertEqual(total, 80)
        self.assertEqual(counts, {0: 40, 1: 40})
        self.assertEqual(first_shape, (3, 3))
        self.assertEqual(batch_sizes, [18, 18, 18, 18, 8])
        self.assertEqual(len(loader_dataset), 5)

    def test_shuffle_is_seed_determined(self) -> None:
        signal, background = self._jets(20, 1.0), self._jets(20, 2.0)
        first = self.dataset(signal, background, seed=0)._labels
        again = self.dataset(signal, background, seed=0)._labels
        other = self.dataset(signal, background, seed=1)._labels
        onp.testing.assert_array_equal(first, again)
        self.assertFalse(onp.array_equal(first, other))

    def test_reader_stops_at_the_requested_count(self) -> None:
        dataset = self.dataset(self._jets(30, 1.0), self._jets(30, 2.0),
                               n_signal=7, n_background=5, batch_size=4)
        self.assertEqual(len(dataset._labels), 12)
        self.assertEqual(int(onp.sum(dataset._labels == 1)), 7)


class TestJetsGameScaling(unittest.TestCase):
    """JetsGame's wider jet cone needs its own assumed range, not JetClass's."""

    def test_assumed_limits_track_the_jet_radius(self) -> None:
        self.assertEqual(jr.JetsGameDataset.ASSUMED_LIMITS['eta'], [-1.0, 1.0])
        self.assertEqual(jr.JetsGameDataset.ASSUMED_LIMITS['phi'], [-1.0, 1.0])
        self.assertEqual(ut.assumed_limits['eta'], [-0.8, 0.8])


class TestDatasetDispatch(_JetsGameFixture):
    """helpers.data must route each configured dataset to its own layout and reader."""

    def _cfg(self, **overrides):
        cfg = OmegaConf.create({'dataset': 'jetsgame', 'data_dir': str(self.root),
                                'signal': 'Top', 'background': 'QCD', 'flat': False,
                                'norm_pt': True, 'jetsgame_pt_bin': '500GeV',
                                'jetsgame_level': 'truth'})
        return OmegaConf.merge(cfg, overrides)

    def test_split_labels_per_dataset(self) -> None:
        jetsgame = self._cfg()
        self.assertEqual([data_helpers.split_label(jetsgame, s) for s in ('train', 'val', 'test')],
                         ['train', 'valid', 'test'])
        jetclass = self._cfg(dataset='jetclass', flat=True)
        self.assertEqual([data_helpers.split_label(jetclass, s) for s in ('train', 'val', 'test')],
                         ['flat_train', 'flat_val', 'flat_test'])

    def test_missing_configs_default_to_jetclass(self) -> None:
        cfg = OmegaConf.create({'data_dir': '/data', 'flat': False})
        self.assertEqual(data_helpers.dataset_name(cfg), 'jetclass')
        with self.assertRaisesRegex(ValueError, "dataset='delphes' is not supported"):
            data_helpers.dataset_name(OmegaConf.create({'dataset': 'delphes'}))

    def test_require_files_names_the_missing_split(self) -> None:
        write_jets(self.path('train', 'Top'), [[massless(300.0, 0.1, 0.2)] * 3])
        write_jets(self.path('train', 'QCD'), [[massless(300.0, 0.1, 0.2)] * 3])
        data_helpers.require_files(self._cfg(), ('train',))  # present: does not raise
        with self.assertRaisesRegex(FileNotFoundError, "Missing JetsGame files.*split 'valid'"):
            data_helpers.require_files(self._cfg(), ('train', 'val'))

    def test_build_loader_uses_the_jetsgame_reader(self) -> None:
        jets = [[massless(300.0, 0.1, 0.2), massless(120.0, -0.2, 0.1), massless(60.0, 0.3, 0.0)]]
        write_jets(self.path('train', 'Top'), jets * 4)
        write_jets(self.path('train', 'QCD'), jets * 4)
        loader = data_helpers.build_loader(self._cfg(), 'train', 4, 4, 3,
                                           batch_size=3, train=True, seed=0)
        self.assertIsInstance(loader.dataset, jr.JetsGameDataset)
        self.assertEqual([data.shape[0] for data, _ in loader], [3, 3, 2])


@unittest.skipUnless(
    os.path.isfile(os.path.join(_JETSGAME_DIR, 'test', 'test_Top_500GeV.json.gz'))
    and os.path.isfile(os.path.join(_JETSGAME_DIR, 'test', 'test_QCD_500GeV.json.gz')),
    "JetsGame data not available",
)
class TestRealJetsGameData(unittest.TestCase):
    """The real distribution must load and land inside the circuit's feature ranges."""

    def test_reads_real_files_into_circuit_ranges(self) -> None:
        loader = jr.JetsGameLoader(
            signal_filelist=jr.jetsgame_files(_JETSGAME_DIR, 'test', 'Top'),
            background_filelist=jr.jetsgame_files(_JETSGAME_DIR, 'test', 'QCD'),
            n_signal=40, n_background=40, batch_size=18,
            input_shape=(10, 3), train=False, normalize_pt=True, seed=0,
        )
        counts, total = {0: 0, 1: 0}, 0
        for data, labels in loader:
            self.assertEqual(tuple(data.shape[1:]), (10, 3))
            total += data.shape[0]
            for label in labels.tolist():
                counts[int(label)] += 1
        self.assertEqual((total, counts), (80, {0: 40, 1: 40}))

        values = onp.asarray(loader.dataset._data)
        for name, index in (('eta', _ETA), ('phi', _PHI)):
            with self.subTest(feature=name):
                self.assertGreaterEqual(values[..., index].min(), -onp.pi)
                self.assertLess(values[..., index].max(), onp.pi)
        self.assertGreaterEqual(values[..., _PT].min(), 0.0)
        self.assertLessEqual(values[..., _PT].max(), 1.0)


if __name__ == '__main__':
    unittest.main()
