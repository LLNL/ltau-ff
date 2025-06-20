# LTAU
LTAU (**L**oss **T**rajectory **A**nalysis for **U**ncertainty) is a technique for estimating the uncertainty in a model's predictions for any regression task
by approximating the probability distribution functions (PDFs) of the model's per-sample errors
over the course of training. The approximated PDFs, combined with a similarity
search algorithm in the model's descriptor space, can then be used to estimate the
model's uncertainty on any given test point.

## LTAU-FF
LTAU-FF (LTAU for atomistic **F**orce
**F**ields) is the application of LTAU spefically for use with atomistic force fields. LTAU-FF has a number of advantages including:
* **Simplicity**: just log per-sample errors at every epoch!
* **Speed**: leveraging the [FAISS library](https://github.com/facebookresearch/faiss) for fast similarity search,
    the computational cost of LTAU-FF can be made negligible compared to a forward pass
    of the model.
* **Utility**: in our published work, we show that LTAU-FF can be used in practical applications for:
  * Generating a well-calibrated UQ metric
  * Detecting out-of-domain data
  * Predicting failure in the OC20 IS2RS task

When publishing results using this package, please cite:

```
@misc{2402.00853,
    Author = {Joshua A. Vita and Amit Samanta and Fei Zhou and Vincenzo Lordi},
    Title = {LTAU-FF: Loss Trajectory Analysis for Uncertainty in Atomistic Force Fields},
    Year = {2024},
    Eprint = {arXiv:2402.00853},
}
```

# Getting started

## Installation

The only dependency of LTAU-FF is the FAISS package.
We recommend first trying:

```bash
git clone https://github.com/LLNL/ltau-ff/ltau-ff.git
cd ltau-ff
pip install -e .
```

which will attempt to install `faiss-cpu==1.8.0`. If this fails (which it will on POWER9 architectures), then instead try:

```bash
conda install faiss-cpu=1.8.0
```

If you encounter further installation difficulties with FAISS, please refer to the instructions provided by [the FAISS documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

Additional software patches are provided for external packages. These do not need to be installed unless you want to use the corresponding functionality.
* [ase_trajectory_patch.py](https://github.com/LLNL/ltau-ff/blob/main/scripts/ase_trajectory_patch.py): for allowing ASE `Trajectory` objects to write non-standard properties (i.e., 'uq', and 'descriptors'). Replaces [ase.io.trajectory](https://gitlab.com/ase/ase/-/blob/master/ase/io/trajectory.py?ref_type=heads). Also requires commenting out this line in [ase.calculators.singlepoint](https://gitlab.com/ase/ase/-/blob/master/ase/calculators/singlepoint.py?ref_type=heads#L25).
* nequip
    * (only needed for nequip < v0.7.0) [nequip_trainer_patch.py](https://github.com/LLNL/ltau-ff/blob/main/scripts/nequip_trainer_patch.py): for logging per-sample energy/force errors at every epoch while training a NequIP model. Replaces [nequip.train.trainer](https://github.com/mir-group/nequip/blob/main/nequip/train/trainer.py).
    * [nequip_modules.py](https://github.com/LLNL/ltau-ff/blob/main/scripts/nequip_modules.py): custom trainer and dataset modules which allow for logging per-atom errors at each epoch. Currently only tested on 1 GPU. You can use these when training your nequip-style model by using these blocks in your config file:
    ```
    training_module:
        _target_: ltau_nequip_modules.LTAULightningModule
        error_save_dir: ${hydra:runtime.output_dir}/${trainer.logger.name}
    data:
        _target_: ltau_nequip_modules.ASEDataModuleWithID 
    ```

Depending on demand, these patches may eventually be opened as pull requests on their respective repositories.

## Examples
* [Basic tutorial](https://github.com/LLNL/ltau-ff/blob/main/examples/tutorial.ipynb)

## Supported models
The [UQEstimator](https://github.com/LLNL/ltau-ff/blob/main/ltau_ff/uq_estimator.py) class is model-agnostic, and will work immediately as long as you can provide the training loss trajectories and the per-sample descriptors.

Some additionally functionality is provided specifically for the NequIP and MACE models; in particular, these include an ASE wrapper for running simulations with UQ, and helper scripts for extracting descriptors and performing energy minimization using a trained model. If you would like to add similar suppport for another model, we recommend taking a look at the following files for reference:

* ASE wrappers:
    * [NequIPUQWrapper](https://github.com/LLNL/ltau-ff/blob/main/ltau_ff/ase_wrapper.py)
    * [MACEUQWrapper](https://github.com/LLNL/ltau-ff/blob/main/ltau_ff/ase_wrapper_mace.py)
* Descriptor extraction:
    * [ltau-ff-nequip-descriptors](https://github.com/LLNL/ltau-ff/blob/main/scripts/ltau-ff-nequip-descriptors)
    * [ltau-ff-nequip-v0.7-descriptors](https://github.com/LLNL/ltau-ff/blob/main/scripts/ltau-ff-nequip-v0.7-descriptors) (for use with `nequip >= 0.7.0`)
    * [ltau-ff-mace-descriptors](https://github.com/LLNL/ltau-ff/blob/main/scripts/ltau-ff-mace-descriptors)
* Energy minimization
    * [ltau-ff-nequip-minimizer](https://github.com/LLNL/ltau-ff/blob/main/scripts/ltau-ff-nequip-minimizer)
    * (not implemented for MACE yet)

# FAQ
* **Can LTAU be used with pre-trained models?**
    * Yes, but you still need to have a method for sampling an ensemble of model parameters. One possibility would be to run Monte Carlo sampling, perturbing the model parameters and logging the per-atom errors. This is the approach taken in ["Bayesian Ensemble Approach to Error Estimation of Interatomic Potentials" by Frederiksen et. al](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.93.165501).

# Contact
If you have any questions or comments, please either open an issue or email Josh
Vita (vita1@llnl.gov)
