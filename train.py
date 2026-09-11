"""
Train a quantum classifier and save its resolved configuration and final weights.

Author: Aritra Bal (ETP)
Date: 2026-09-10
"""

from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import datetime
import glob
import time
import matplotlib.pyplot as plt
import numpy as nnp
import pennylane.numpy as np
import pennylane as qml
import helpers.utils as ut
import case_reader as cr
import quantum.losses as loss
import quantum.architectures as qc
from quantum.circuits.base import CircuitWeights
from helpers.config import (
    load_config,
    run_directory,
    save_config,
    validate_training_config,
)
from helpers.trained_run import (
    implementation_signature,
    load_circuit_snapshot,
    load_training_checkpoint,
    resolve_aux_weights,
    save_circuit_snapshot,
    save_trained_run,
    validate_weights,
)
from loguru import logger
from dotenv import load_dotenv
import wandb

load_dotenv()


def main(cfg: DictConfig):
    """Train with resolved run settings and certify successfully completed weights."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cfg.seed = str(cfg.seed)
    resume = cfg.resume
    random_seed = validate_training_config(cfg, require_random_seed=not resume)
    save_dir = str(run_directory(cfg.save_dir, cfg.seed, random_seed))
    if resume:
        cfg = OmegaConf.load(os.path.join(save_dir, 'config.yaml'))
        OmegaConf.resolve(cfg)
        random_seed = validate_training_config(cfg)
        logger.warning('Resume uses the saved configuration; other training-option overrides are ignored.')
    else:
        for key in ('save_dir', 'data_dir', 'dump'):
            cfg[key] = os.path.abspath(os.path.expanduser(cfg[key]))
        cfg.seed = str(cfg.seed)
        save_dir = str(run_directory(cfg.save_dir, cfg.seed, random_seed))
        legacy_config = run_directory(cfg.save_dir, cfg.seed) / 'config.yaml'
        if legacy_config.exists():
            raise FileExistsError(
                f'Legacy run already uses experiment directory {legacy_config.parent}.'
            )
        try:
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f'Run already exists at {save_dir}; choose a new seed or resume it.'
            ) from error
    # Set up directories
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    run_str = str(cfg.seed)
    if random_seed is not None:
        run_str = f"{run_str}_{random_seed}"

    wandb_config = OmegaConf.to_container(cfg)
    for path_key in ('data_dir', 'save_dir', 'dump'):
        wandb_config.pop(path_key, None)
    wandb.init(
        entity="aritrabal2-Karlsruhe Institute of Technology",
        project="neo1P1Q",
        config=wandb_config,
        name=run_str,
        notes=cfg.desc,
    )
    with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb.run.id)

    # Logging setup
    logger.add(
        os.path.join(save_dir, 'logs.log'), rotation='10 MB', backtrace=True,
        diagnose=True, level='DEBUG', mode='a' if resume else 'w',
    )
    logger.info("########################################### \n\n")
    logger.info(f"This circuit contains {cfg.wires} qubits")
    
    print("Will save models to: ", save_dir)

    # Preserve the exact config this run used, in one reusable file.
    if not resume:
        save_config(cfg, os.path.join(save_dir, 'config.yaml'))
    circuit_dir = pathlib.Path(save_dir) / 'circuits'
    if not resume:
        save_circuit_snapshot(save_dir)
    circuit_registry = load_circuit_snapshot(save_dir)
    signature = implementation_signature(circuit_dir)

    # Further setup based on config
    if resume:
        checkpoint_path, checkpoint = load_training_checkpoint(save_dir, cfg, signature)
        init_weights = checkpoint['weights']

        logger.info("########################################### \n\n")
        logger.info(f"Resuming training from last checkpoint at {checkpoint_path}")
        logger.info(f"Using arguments specified in original training run")
        logger.info(f"Training resumed at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info("Current weights are: ", init_weights)
        logger.info("\n\n ########################################### \n\n")
    else:
        logger.info("########################################### \n\n")
        logger.info(f"This circuit contains {cfg.wires} qubits")
        logger.info("\n\n ########################################### \n\n")
        
    logger.info(f"Feature are scaled to the following limits: {ut.feature_limits}")

    if cfg.norm_pt:
        logger.info(f"pT will not be scaled to the above limit. Will be normalized using 1/jet_pt")
    else:
        logger.info(f"pT will also be scaled assuming above maxima")
    if cfg.flat:
        logger.info("Using flat mjj distribution for training")
    cost_fn = loss.VQC_cost
    if cfg.shots < 0:
        logger.warning("Negative shots specified. Setting to None for an analytic calculation.")
    # Create the quantum classifier instance
    VQC = qc.QuantumClassifier.from_config(cfg, circuit_registry=circuit_registry)

    if not resume:
        shape = VQC._impl.rotation_shape(len(VQC.auto_wires), cfg.num_layers)
        rng = nnp.random.default_rng(random_seed)
        rot = np.array(
            rng.uniform(0, np.pi, size=(shape.L, shape.N, shape.R)),
            requires_grad=True,
        )
        aux = resolve_aux_weights(VQC, dict(cfg.get('aux_weights', {})))
        aux = {k: np.array(v, requires_grad=True) for k, v in aux.items()}
        init_weights = CircuitWeights(rot=rot, aux=aux)
    validate_weights(VQC, init_weights, cfg)

    train_max_n = cfg.n_signal + cfg.n_background
    valid_max_n = cfg.n_signal_val + cfg.n_background_val
    
    # Print training parameters using the VQC instance method
    VQC.print_training_params()

    # Load the data: balanced signal vs background, per the config keys.
    # JetClass layout is <data_dir>/<split>/<sample>/<sample>_NNN.h5.
    train_split = 'flat_train' if cfg.flat else 'train'
    val_split = 'flat_val' if cfg.flat else 'val'
    _class_files = lambda split, sample: sorted(glob.glob(os.path.join(cfg.data_dir, split, sample, '*.h5')))
    train_sig, train_bg = _class_files(train_split, cfg.signal), _class_files(train_split, cfg.background)
    val_sig, val_bg = _class_files(val_split, cfg.signal), _class_files(val_split, cfg.background)
    required_particles = len(VQC.auto_wires) * VQC.num_layers
    num_particles = getattr(cfg, 'num_particles', required_particles)
    if num_particles < required_particles:
        raise ValueError(
            f"The {cfg.num_layers}-layer circuit requires at least "
            f"{required_particles} particles, but num_particles={num_particles}"
        )
    logger.info(f"Number of particles to load: {num_particles}")

    if not (train_sig and train_bg and val_sig and val_bg):
        raise FileNotFoundError(
            f"Missing JetClass files under {cfg.data_dir} for signal='{cfg.signal}', "
            f"background='{cfg.background}' (splits '{train_split}', '{val_split}')"
        )
    logger.info(f"Training set: {cfg.n_signal} '{cfg.signal}' + {cfg.n_background} '{cfg.background}' jets from {train_split}/")
    logger.info(f"Validation set: {cfg.n_signal_val} '{cfg.signal}' + {cfg.n_background_val} '{cfg.background}' jets from {val_split}/")

    train_loader = cr.OneP1QDataLoader(
        signal_filelist=train_sig, background_filelist=train_bg,
        n_signal=cfg.n_signal, n_background=cfg.n_background,
        batch_size=cfg.batch_size,
        input_shape=(num_particles, 3),
        train=True,
        normalize_pt=cfg.norm_pt,
        logger=logger,
        seed=random_seed if random_seed is not None else 0,
    )
    val_loader = cr.OneP1QDataLoader(
        signal_filelist=val_sig, background_filelist=val_bg,
        n_signal=cfg.n_signal_val, n_background=cfg.n_background_val,
        batch_size=cfg.batch_size,
        input_shape=(num_particles, 3),
        train=False,
        normalize_pt=cfg.norm_pt,
        logger=logger,
        seed=random_seed if random_seed is not None else 0,
    )

    # Initialize Adam with either the configured settings or the saved settings.
    # The jax backend always builds the same wrapped-adam transformation --
    # optax.inject_hyperparams keeps the learning rate inside opt_state itself,
    # and restore_checkpoint() overwrites that state wholesale on resume, so
    # there is nothing backend-specific to read from the checkpoint here.
    if cfg.backend == 'jax':
        import optax
        optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=cfg.lr)
    elif resume:
        optimizer_state = checkpoint['optimizer']
        optimizer = qml.AdamOptimizer(
            stepsize=optimizer_state['stepsize'],
            beta1=optimizer_state['beta1'],
            beta2=optimizer_state['beta2'],
            eps=optimizer_state['eps'],
        )
    else:
        optimizer = qml.AdamOptimizer(stepsize=cfg.lr)

    # Initialize the trainer with the new class signature
    trainer = qc.QuantumTrainer(
        model=VQC,
        lr=cfg.lr,
        optimizer=optimizer,
        loss_fn=cost_fn,
        save=cfg.save,
        train_max_n=train_max_n,
        valid_max_n=valid_max_n,
        epochs=cfg.epochs,
        improv=cfg.improv,
        min_epochs=cfg.min_epochs,
        decay_rate=cfg.decay_rate,
        decay_patience=cfg.decay_patience,
        wandb=wandb,
        loss_type=cfg.loss,
        checkpoint_config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        circuit_signature=signature,
        # Pass additional parameters via kwargs
        init_weights=init_weights,
        batch_size=cfg.batch_size,
        logger=logger
    )

    trainer.print_params('Initialized parameters!')
    trainer.set_directories(save_dir)

    if resume:
        trainer.restore_checkpoint(checkpoint)
        logger.info(f"Resuming training from epoch {trainer.current_epoch}")
    else:
        logger.info(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info(f'Epochs: {cfg.epochs} | Learning rate: {cfg.lr} | Batch size: {cfg.batch_size} \nBackend: {cfg.backend} | Wires: {cfg.wires} | Shots: {cfg.shots} \n')    
        logger.info(f'Additional information: {cfg.desc}')

    if cfg.get('evictable', False):
        evictable_id = cfg.seed if random_seed is None else f'{cfg.seed}/{random_seed}'
        trainer.is_evictable_job(seed=evictable_id)

    # Begin training
    abs_start = time.time()
    try:
        history = trainer.run_training_loop(train_loader, val_loader)
        if cfg.save:
            save_trained_run(save_dir, cfg, VQC, trainer.current_weights, history, signature)
    except KeyboardInterrupt:
        print("WHYYYYY")
        print("DON'T PRESS CTRL+C AGAIN. I'M TRYING TO SAVE THE CURRENT MODEL AND WRITE TO LOG!")
        ut.Pickle({'weights': trainer.current_weights}, 'aborted_weights.pickle', path=save_dir)
        trainer.print_params('Training aborted. Current parameters are: ')
        raise
    finally:
        logger.info('Training completed with the following parameters:')
        trainer.print_params('Trained parameters:')
        history = trainer.fetch_history()
        print(history)
        if cfg.save:
            done_epochs = len(history['train'])
            ut.Pickle(history, 'history', path=save_dir)
            fig, axes = plt.subplots(figsize=(15, 12))
            axes.plot(np.arange(done_epochs), history['train'], label='train', linewidth=2)
            axes.plot(np.arange(done_epochs + 1), history['val'], label='val', linewidth=2)
            axes.set_xlabel('Epochs', size=25)
            axes.set_ylabel('$1-<T|F> $(in %)', size=25)
            axes.set_xticks(np.arange(0, done_epochs + 1, 5))
            axes.legend(prop={'size': 25})

            axes.tick_params(labelsize=20)
            fig.savefig(os.path.join(save_dir, 'history'))
            abs_end = time.time()
            logger.info(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
            logger.info(f"Total time taken including all overheads: {abs_end - abs_start:.2f} seconds")

        # Close WandB run
        wandb.finish()


if __name__ == "__main__":
    main(load_config())
