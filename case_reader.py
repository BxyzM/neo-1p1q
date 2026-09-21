"""
Load and batch the Monte Carlo datasets used for the CMS EXO-22-026 studies.

Author: Aritra Bal (ETP)
Date: 2026-09-09
"""

import os
import h5py
import helpers.utils as ut
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
import __main__
import numpy as nnp
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Tuple, Union
import loguru
eta_lims=ut.feature_limits['eta']
phi_lims=ut.feature_limits['phi']
class CASEJetClassDataset(IterableDataset):
    """
    Balanced binary dataset built from JetClass-style .h5 files.

    Reads `n_signal` jets from the signal files (label 1) and `n_background`
    jets from the background files (label 0) -- stopping part-way through a file
    once a target is met -- then concatenates and shuffles the two classes.
    Each .h5 file holds `jetConstituentsList` of shape (N, 100, 3): N jets, up
    to 100 particles each, (eta, phi, pt) per particle.

    Args:
        signal_filelist (List[str]): .h5 files for the signal class (label 1).
        background_filelist (List[str]): .h5 files for the background class (label 0).
        n_signal (int): number of signal jets to read.
        n_background (int): number of background jets to read.
        batch_size (int): jets per yielded batch.
        data_key (str): key for the per-particle data inside the .h5 files.
        feature_key (str): key for jet-level features.
        input_shape (tuple[int]): (particles per jet to keep, features per particle).
        epsilon (float): small offset used by the fixed rescaling.
        train (bool): kept for call-site compatibility; both classes are always labelled.
        normalize_pt (bool): divide particle pt by jet pt instead of fixed rescaling.
        logger: optional loguru-style logger; falls back to print.
        seed (int): RNG seed for the signal/background shuffle.

    Yields:
        Tuple[np.ndarray, np.ndarray]: a batch of jets and their integer labels.
    """
    #: Name used in this reader's log lines; subclasses reading another dataset override it.
    LOADER_NAME = 'JetClass'
    #: Source ranges fed to fixed_rescale. Overridden per dataset (see helpers.utils).
    ASSUMED_LIMITS = ut.assumed_limits

    def __init__(self, signal_filelist:List[str], background_filelist:List[str],
                 n_signal:int, n_background:int, batch_size:int=32,
                 data_key='jetConstituentsList', feature_key='jetFeatures',
                 input_shape:tuple[int]=(10, 3), epsilon:float=1.0e-4, train:bool=True,
                 normalize_pt:bool=False, logger=None, seed:int=0):
        super().__init__()
        self.signal_filelist = sorted(signal_filelist)
        self.background_filelist = sorted(background_filelist)
        self.n_signal = int(n_signal)
        self.n_background = int(n_background)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pt_index=ut.getIndex('particle','pt')
        self.eta_index=ut.getIndex('particle','eta')
        self.phi_index=ut.getIndex('particle','phi')
        self.jpt_index=ut.getIndex('jet','jet_pt')
        self.epsilon=epsilon
        self.data_key=data_key
        self.feature_key=feature_key
        self.train=train
        self.normalize_pt=normalize_pt
        self.logger=logger
        self.n_qubits=input_shape[0]
        self.seed=seed
        self._data, self._labels = self._materialise()

    def _log(self, msg:str) -> None:
        """Route a message through the logger if present, else stdout."""
        self.logger.info(msg) if self.logger is not None else print(msg)

    def fixed_rescale(self,data: np.ndarray, epsilon: float = 1.0e-4, type='pt') -> np.ndarray:
        """
        Rescales the data to a specified range. Instead of using the min/max values of the data array,
        uses fixed values instead, which can be modified in the assumed_limits array

        Args:
            data (np.ndarray): Input data to rescale.
            epsilon (float): Small offset to prevent numerical instability.

        Returns:
            np.ndarray: Rescaled data.
        """
        min=ut.feature_limits[type]['min']
        max=ut.feature_limits[type]['max']
        max-=epsilon
        assumed_limits=self.ASSUMED_LIMITS
        if type not in ['pt','eta','phi']:
            raise NameError("Type must be either of [pt,eta,phi]")
        if self.logger is not None:
            self.logger.info(f"Assuming fixed sample maxima: [{assumed_limits[type][0]},{assumed_limits[type][1]}] for variable {type}")
        else:
            print(f"Assuming fixed sample maxima: [{assumed_limits[type][0]},{assumed_limits[type][1]}] for variable {type}")
        data_shape = data.shape
        data_reshaped = data.flatten()
        data_scaled = ((data_reshaped - assumed_limits[type][0])/(assumed_limits[type][1]-assumed_limits[type][0]))*(max-min) + min # scale using fixed values of 
        return data_scaled.reshape(data_shape[0], data_shape[1])
    
    def _load_file(self, file_path:str, max_rows:int) -> np.ndarray:
        """
        Load one .h5 file and return its preprocessed jets.

        Keeps the `n_qubits` hardest particles per jet, then rescales pt/eta/phi
        -- either dividing pt by jet pt (`normalize_pt`) or applying the fixed
        rescaling from helpers.utils. JetClass stores constituents already in
        descending-pt order, so the hardest particles are just the leading slice;
        this is asserted so a non-sorted file fails loudly instead of silently
        feeding the wrong particles.

        Args:
            file_path: HDF5 file to read.
            max_rows: Maximum number of leading jets to read.

        Returns:
            np.ndarray: shape (M, n_qubits, 3).
        """
        rows = slice(0, max_rows)
        with h5py.File(file_path, 'r') as file:
            jet_pt = np.array(file[self.feature_key][rows, self.jpt_index])
            jet_etaphipt = np.array(file[self.data_key][rows])
        pt_col = jet_etaphipt[..., self.pt_index]
        if not nnp.all(nnp.diff(pt_col, axis=-1) <= 1.0e-3):
            raise ValueError(
                f"{file_path}: constituents are not in descending-pt order; "
                f"the leading-slice selection assumes they are"
            )
        jet_etaphipt = jet_etaphipt[:, :self.n_qubits, :]

        if self.normalize_pt:
            jet_etaphipt[..., self.pt_index] = jet_etaphipt[..., self.pt_index] / jet_pt[:, np.newaxis]
        else:
            jet_etaphipt[..., self.pt_index] = self.fixed_rescale(jet_etaphipt[..., self.pt_index], epsilon=self.epsilon, type='pt')
        jet_etaphipt[..., self.eta_index] = self.fixed_rescale(jet_etaphipt[..., self.eta_index], epsilon=self.epsilon, type='eta')
        jet_etaphipt[..., self.phi_index] = self.fixed_rescale(jet_etaphipt[..., self.phi_index], epsilon=self.epsilon, type='phi')
        return jet_etaphipt

    def _class_name(self, filelist:List[str]) -> str:
        """Label for this class in log lines; JetClass keeps one sample per directory."""
        return os.path.basename(os.path.dirname(filelist[0])) if filelist else '?'

    def _read_class(self, filelist:List[str], n_target:int, label:int) -> Tuple[np.ndarray, np.ndarray]:
        """Read up to `n_target` jets across `filelist`, all tagged with `label`."""
        name = self._class_name(filelist)
        self._log(f"Reading up to {n_target} '{name}' jets (label {label}); keeping the {self.n_qubits} hardest particles each")
        chunks, count = [], 0
        for file_path in filelist:
            if count >= n_target:
                break
            jets = self._load_file(file_path, n_target - count)
            chunks.append(jets)
            count += len(jets)
        if count < n_target:
            self._log(f"WARNING: requested {n_target} '{name}' jets but only {count} were available")
        data = np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.n_qubits, 3))
        return data, nnp.full(len(data), label, dtype=int)

    def _materialise(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read both classes, concatenate, shuffle, and log the resulting split."""
        sig_data, sig_labels = self._read_class(self.signal_filelist, self.n_signal, 1)
        bg_data, bg_labels = self._read_class(self.background_filelist, self.n_background, 0)
        data = np.concatenate([sig_data, bg_data], axis=0)
        labels = nnp.concatenate([sig_labels, bg_labels], axis=0)

        idx = nnp.random.default_rng(self.seed).permutation(len(data))
        data, labels = data[idx], labels[idx]

        n_sig, n_bg = len(sig_data), len(bg_data)
        total = n_sig + n_bg
        self._log(f"{self.LOADER_NAME} loader finished: {total} jets read -- {n_sig} signal (label 1), {n_bg} background (label 0)")
        if total:
            self._log(
                f"  post-rescale ranges: pt [{float(data[..., self.pt_index].min()):.3f}, {float(data[..., self.pt_index].max()):.3f}], "
                f"eta [{float(data[..., self.eta_index].min()):.3f}, {float(data[..., self.eta_index].max()):.3f}], "
                f"phi [{float(data[..., self.phi_index].min()):.3f}, {float(data[..., self.phi_index].max()):.3f}]"
            )
        return data, labels

    def __iter__(self)-> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        '''Yield the pre-loaded, pre-shuffled jets in batches of `batch_size`.'''
        for i in range(0, len(self._data), self.batch_size):
            batch_data = torch.from_numpy(nnp.asarray(self._data[i:i + self.batch_size])).float()
            batch_labels = self._labels[i:i + self.batch_size]
            yield np.array(batch_data, requires_grad=False), np.array(batch_labels, dtype=int, requires_grad=False)

    def __len__(self) -> int:
        """Return the number of batches yielded, including a partial final batch."""
        return (len(self._data) + self.batch_size - 1) // self.batch_size


def OneP1QDataLoader(input_shape:tuple[int]=(100, 3),train:bool=True,**kwargs) -> DataLoader:
    '''
    Build a DataLoader over a balanced signal/background CASEJetClassDataset.

    Pass `signal_filelist`, `background_filelist`, `n_signal`, `n_background`
    (plus optional `batch_size`, `normalize_pt`, `logger`, `seed`) via kwargs.

    Returns:
        DataLoader: yields (jets, labels) batches; batching is done by the dataset.
    '''
    print(f"Will read only {input_shape[0]} particles per jet")
    dset = CASEJetClassDataset(input_shape=input_shape,train=train,**kwargs)
    return DataLoader(dset, batch_size=None)  # None for batch_size since batching is managed by the dataset


def select_subjet_constituents(jet_etaphipt, num_PFCands_subleading_jet, evt_subjet_idx, n_qubits=8,selection='equal'):
    """
    Selects a fixed number of particles per jet, filling with random sampling if needed.

    Parameters:
    - num_PFCands_subleading_jet: (N,) array representing number of particles in the lower pT subjet.
    - evt_subjet_idx: (N, 100) array containing indices of particles assigned to subjets.
    - jet_etaphipt: (N, 100, 3) array representing momenta (eta, phi, pT) of particles.
    - n_qubits: Desired number of particles per jet (default is 8).

    Returns:
    - jet_etaphipt_selected: (N, n_qubits, 3) array of selected particles per jet.
    """
    if n_qubits%2!=0:
        raise ValueError("No. of particles to select = no. of qubits must be even")
    N = jet_etaphipt.shape[0]  # Number of jets

    # Step 1: Create mask limit to determine how many particles can be selected from each subjet
    if selection=='random':
        mask_limit = (n_qubits//2)*np.ones([N,1])#np.where(num_PFCands_subleading_jet > n_qubits // 2, n_qubits // 2, n_qubits - num_PFCands_subleading_jet)  # (N,)
        pf_mask = evt_subjet_idx < mask_limit  # (N, 100)
        
    # Step 3: Ensure each jet has exactly n_qubits particles
        selected_particles = []
        print("It is now necessary to run an event loop in python")
        print("My sincere apologies")
        import time;time.sleep(3)
        for i in range(N):
            jet_particles = jet_etaphipt[i]  # (100, 3)
            mask = pf_mask[i]  # (100,)
            selected = jet_particles[mask]  # Select particles based on mask
            selected_idx=evt_subjet_idx[i][mask]
            if len(selected) < n_qubits:
                # Randomly sample additional particles to reach n_qubits
                non_padded_indices = np.where(~mask)[0]  # Indices of non-padded particles
                num_additional = n_qubits - len(selected)
                additional_indices = np.random.choice(non_padded_indices, size=num_additional, replace=False)
                additional_particles = jet_particles[additional_indices]
                selected = np.vstack((selected, additional_particles))

            selected_particles.append(selected)

        # Step 4: Convert the list to a numpy array of shape (N, n_qubits, 3)
        jet_etaphipt = np.array(selected_particles)  # (N, n_qubits, 3)
        return jet_etaphipt
    elif selection=='equal':
        mask_limit = np.where(num_PFCands_subleading_jet > n_qubits // 2, n_qubits // 2, n_qubits - num_PFCands_subleading_jet)  # (N,)
        pf_mask=evt_subjet_idx<mask_limit
        extra_mask=np.sum(pf_mask,axis=1)==n_qubits
        
        jet_etaphipt=jet_etaphipt[extra_mask] # N x n_qubits x 3 
        pf_mask=pf_mask[extra_mask]
        jet_etaphipt_selected=jet_etaphipt[pf_mask].reshape(-1,n_qubits,3) # N x 2 x n_qubits x 3
        return jet_etaphipt_selected,extra_mask    
    else:
        raise NameError("Selection must be either random or equal")
    # Step 2: Create a mask to select particles from each jet
