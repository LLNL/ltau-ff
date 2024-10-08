import numpy as np

from nequip.ase import NequIPCalculator
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from nequip.data import AtomicData, AtomicDataDict


class NequIPUQWrapper(NequIPCalculator):
    """A wrapper to a NequIP model which optionally returns the errors on the
    force predictions of the model. See nequip.ase.NequIPCalculator for
    additional documentation.
    """

    implemented_properties = NequIPCalculator.implemented_properties + ['uq', 'descriptors']

    def __init__(
        self,
        model,
        r_max,
        uq_estimator,
        device,
        *args,
        **kwargs
        ):
        """
        Args:
            model (NequIP): the NequIP model
            r_max (float): the radial cutoff of the model
            uq_estimator (UQEstimator): the UQ estimator
        """
        NequIPCalculator.__init__(self, model, r_max=r_max, device=device, *args, **kwargs)

        self.uq_estimator = uq_estimator


    def _parse_data(self, atoms):
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        return data


    def get_descriptors(self, atoms):
        data = self._parse_data(atoms)

        out = self.model(data)

        descriptors = out[AtomicDataDict.NODE_FEATURES_KEY]
        descriptors = descriptors.detach().cpu().numpy()

        return descriptors


    # @profile
    def calculate(
            self,
            atoms=None,
            properties=["energy"],
            system_changes=all_changes,
            uq=True,
            topk=10,
            atol=None,
            ):
        """
        Calculate properties.

        This is copy-pasted from the NequIP source code, with the exception of
        the UQ parts. This was done to avoid having to do two forward passes
        (one to calculate the property, and one to extract the descriptors for
        UQ).

        atoms (ase.Atoms): the atoms object
        properties (list[str]): properties to be computed, used by ASE
            internally 
        system_changes (list[str]): system changes since last calculation,
            used by ASE internally
        uq (bool): if True, return the UQ metric for the force predictions on
            all atoms.
        topk (int): the number of neighbors to average over use when performing
            UQ. default=10
        atol (float): error threshold at which to compute the uncertainties
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        data = self._parse_data(atoms)

        # predict + extract data
        out = self.model(data)
        self.results = {}
        # only store results the model actually computed to avoid KeyErrors
        if AtomicDataDict.TOTAL_ENERGY_KEY in out:
            self.results["energy"] = self.energy_units_to_eV * (
                out[AtomicDataDict.TOTAL_ENERGY_KEY]
                .detach()
                .cpu()
                .numpy()
                .reshape(tuple())
            )
            # "force consistant" energy
            self.results["free_energy"] = self.results["energy"]
        if AtomicDataDict.PER_ATOM_ENERGY_KEY in out:
            self.results["energies"] = self.energy_units_to_eV * (
                out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
                .detach()
                .squeeze(-1)
                .cpu()
                .numpy()
            )
        if AtomicDataDict.FORCE_KEY in out:
            # force has units eng / len:
            self.results["forces"] = (
                self.energy_units_to_eV / self.length_units_to_A
            ) * out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in out:
            stress = out[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
            stress = stress.reshape(3, 3) * (
                self.energy_units_to_eV / self.length_units_to_A**3
            )
            # ase wants voigt format
            stress_voigt = full_3x3_to_voigt_6_stress(stress)
            self.results["stress"] = stress_voigt

        if uq:
            if self.uq_estimator is None:
                raise RuntimeError("`uq` cannot be True if `uq_estimator` is None")

            descriptors = out[AtomicDataDict.NODE_FEATURES_KEY]
            descriptors = descriptors.detach().cpu().numpy()

            self.results['uq'] = self.uq_estimator.predict_errors(descriptors, topk)
            self.results['descriptors'] = descriptors
