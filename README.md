# LTAU
LTAU (**L**oss **T**rajectory **A**nalysis for **U**ncertainty) is a technique for estimating the uncertainty in a model's predictions for any regression task
by approximating the cumulative distribution function (CDF) of the model's errors
over the course of training. The approximated CDF, combined with a similarity
search algorithm in the model's descriptor space, can then be used to estimate the
likelihood that the model's predictions on any given test point will fall below
a chosen threshold.

## LTAU-FF
LTAU-FF (LTAU in atomistic **F**orce
**F**ields) is the application of LTAU spefically for use with atomistic force fields. LTAU-FF has a number of advantages including:
* **Simplicity**: just log per-sample errors at every epoch!
* **Speed**: leveraging the [FAISS library](https://github.com/facebookresearch/faiss) for fast similarity search,
    the computational cost of LTAU-FF can be made negligible compared to a forward pass
    of the model.
* **Utility**: in our published work, we show that LTAU-FF can be used in practical applications for:
  * Generating an easily-calibrated UQ metric
  * Tuning the training-validation gap
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

The only dependency of LTAU-FF is the FAISS package, which can be installed
following [the instructions provided here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

Once FAISS has been installed, you can install LTAU-FF by running the following commands:

```bash
git clone https://github.com/LLNL/ltau-ff/ltau-ff.git
cd ltau-ff
pip install -e .
```

Additional software patches are provided for external packages. These do not need to be installed unless you want to use the corresponding functionality.
* [ase_trajectory_patch.py](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/scripts/ase_trajectory_patch.py?ref_type=heads): for allowing ASE `Trajectory` objects to write non-standard properties (i.e., 'uq', and 'descriptors'). Replaces [ase.io.trajectory](https://gitlab.com/ase/ase/-/blob/master/ase/io/trajectory.py?ref_type=heads). Also requires commenting out this line in [ase.calculators.singlepoint](https://gitlab.com/ase/ase/-/blob/master/ase/calculators/singlepoint.py?ref_type=heads#L25).
* [nequip_trainer_patch.py](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/scripts/nequip_trainer_patch.py?ref_type=heads): for logging per-sample energy/force errors at every epoch while training a NequIP model. Replaces [nequip.train.trainer](https://github.com/mir-group/nequip/blob/main/nequip/train/trainer.py).

Depending on demand, these patches may eventually be opened as pull requests on their respective repositories.

## Examples
* [Basic tutorial](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/examples/tutorial.ipynb?ref_type=heads)

# Supported models

## Models
The [UQEstimator](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/ltau_ff/uq_estimator.py?ref_type=heads#L5) class is model-agnostic, and will work immediately as long as you can provide the pre-computed uncertainties (or the corresponding training loss trajectories) and the per-sample descriptors.

Some additionally functionality is provided specifically for the NequIP model; in particular, these include an ASE wrapper for running simulations with UQ, and helper scripts for extracting descriptors and performing energy minimization using a trained model. If you would like to add similar suppport for another model, we recommend taking a look at the following files for reference:

* [NequIPUQWrapper](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/ltau_ff/ase_wrapper.py?ref_type=heads#L9): for creating an ASE wrapper with UQ support.
* [ltau-ff-nequip-descriptors](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/scripts/ltau-ff-nequip-descriptors?ref_type=heads): for extracting descriptors using the ASE wrapper.
* [ltau-ff-nequip-minimizer](https://github.com/LLNL/ltau-ff/ltau-ff/-/blob/main/scripts/ltau-ff-nequip-minimizer?ref_type=heads): for performing energy minimization with the ASE wrapper while logging uncertainties



# Contact
If you have any questions or comments, please either open an issue or email Josh
Vita (vita1@llnl.gov)