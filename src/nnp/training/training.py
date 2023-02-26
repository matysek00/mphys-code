#!/usr/bin/env python3

import torch
import torchmetrics
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy


import schnetpack as spk

_fun_activations = {
    'softplus': torch.nn.functional.softplus,
    'tanh': torch.nn.functional.tanh, 
    'sigmoid': torch.nn.functional.sigmoid,
    'silu': torch.nn.functional.silu,
}

def get_trainer(
    device: str, 
    epochs: int, 
    n_cpu: int, 
    use_logger: bool = False, 
    strategy: pl.strategies = DDPStrategy(find_unused_parameters=False)
    ):
    
     
    #logger = pl.loggers.TensorBoardLogger(save_dir='./')
    logger = pl.loggers.CSVLogger(save_dir='./', flush_logs_every_n_steps=1000)

    callbacks = [
        spk.train.ModelCheckpoint(
            model_path="best_inference_model",
            save_top_k=1,
            monitor="val_loss",
            
        )
    ]
    
    trainer = pl.Trainer( 
        logger= logger,# if use_logger else False,
        callbacks=callbacks,
        default_root_dir='.',
        max_epochs=epochs, # for testing, we restrict the number of epochs
        accelerator = device,
        devices=n_cpu,
        #strategy=strategy,
        auto_lr_find = True,
        enable_progress_bar = False,
    )
    
    return trainer


def prepare_data(fn_db, n_cpu, batch_size, transforms):

    # ase enviroment on cpu torch on gpu
    dataset = spk.data.AtomsDataModule(
        fn_db,
        batch_size,
        num_train = .9,
        num_val = .1,
        transforms = transforms,
        num_workers = n_cpu, 
        distance_unit='Ang',
        property_units={'energy':'eV', 'forces':'eV/Ang'},
        pin_memory = False,

    )
    
    dataset.prepare_data()
    dataset.setup()
    return dataset


def get_transforms(dir_nl_cache, cutoff):
    NL_provider = spk.transform.ASENeighborList(cutoff=cutoff)

    # remove caching for slower but but memory efficient perfomance
    transforms = [
        spk.transform.RemoveOffsets('energy', remove_mean = True), # cetner at zero
        spk.transform.CachedNeighborList(dir_nl_cache, NL_provider),
        spk.transform.CastTo32()
    ]
    
    # transform the energy back
    postprocess = [
        spk.transform.AddOffsets('energy', add_mean=True),
        spk.transform.CastTo64()
    ]
    
    return transforms,postprocess


def get_task(cutoff, 
            n_layers, 
            n_hidden, 
            n_interacions, 
            n_features, 
            n_guassians, 
            rho, 
            learning_rate, 
            postprocess, 
            activation
            ):
    # Create the whole model
    #cutoff_module = spk.atomistic.FilterShortRange(cutoff)

    pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms

    radial_basis = spk.nn.GaussianRBF(n_rbf=n_guassians, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_features, n_interactions=n_interacions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )


    cal_activation = _fun_activations[activation]
    pred_energy = spk.atomistic.Atomwise(n_in=n_features, output_key='energy', n_hidden=n_hidden, n_layers=n_layers, activation=cal_activation)
    pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=postprocess
    )

    output_energy = spk.task.ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=rho,
        metrics={
            "MSE": torchmetrics.MeanSquaredError()
        }
    )

    output_forces = spk.task.ModelOutput(
        name='forces',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1-rho,
        metrics={
            "MSE": torchmetrics.MeanSquaredError()
        }
    )
    
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": learning_rate}
    )
    
    return task