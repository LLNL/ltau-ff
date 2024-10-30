import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from typing import Union

from mace.calculators import MACECalculator
from mace.modules.utils import extract_invariant

class MACEUQWrapper(MACECalculator):
    """A wrapper to a NequIP model which optionally returns the errors on the
    force predictions of the model. See nequip.ase.NequIPCalculator for
    additional documentation.
    """

    implemented_properties = MACECalculator.implemented_properties + ['uq']

    def __init__(
        self,
        model_paths: Union[list, str],
        uq_estimator,
        device: str,
        energy_units_to_ev: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        **kwargs,
    ):
        super().__init__(
            model_paths=model_paths,
            device=device,
            energy_units_to_ev=energy_units_to_ev,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            model_type=model_type,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            **kwargs,
        )

        self.uq_estimator = uq_estimator

    # pylint: disable=dangerous-default-value
    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
        uq=False,
        uq_neighbors=1,
        ):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :param uq: [bool], if True, return the UQ metric for the force predictions on all atoms
        :param uq_neighbors: [int], the number of neighbors to use for estimating uncertainty
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = self._clone_batch(batch_base)
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])
            compute_stress = not self.use_compile
        else:
            compute_stress = False

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )

        if uq:
            ret_tensors['uq'] = np.zeros((self.num_models, len(atoms)))

        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            out = model(
                batch.to_dict(),
                compute_stress=compute_stress,
                training=self.use_compile,
            )
            if self.model_type in ["MACE", "EnergyDipoleMACE"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["node_energy"][i] = (out["node_energy"] - node_e0).detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()
                if uq:
                    if self.model_type != 'MACE':
                        raise NotImplementedError("Descriptor exctraction, used"
                        " for UQ, is only implemented for MACE models")

                    # copied code from get_descriptors() to avoid repeating
                    # model forward pass; uses default values
                    # (invariants_only=True, num_layers=-1)
                    descriptors = out['node_feats']
                    irreps_out = self.models[0].products[0].linear.__dict__["irreps_out"]
                    l_max = irreps_out.lmax
                    num_features = irreps_out.dim // (l_max + 1) ** 2
                    descriptors = extract_invariant(
                            descriptors,
                            num_layers=int(self.models[0].num_interactions),
                            num_features=num_features,
                            l_max=l_max,
                        )

                    ret_tensors['uq'][i] = self.uq_estimator.predict_errors(
                        descriptors.detach().cpu().numpy(),
                        uq_neighbors,
                    )

            if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
                ret_tensors["dipole"][i] = out["dipole"].detach()

        self.results = {}
        if self.model_type in ["MACE", "EnergyDipoleMACE"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["node_energy"] = (
                torch.mean(ret_tensors["node_energy"], dim=0).cpu().numpy()
            )
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if uq:
                self.results["uq"] = (
                    np.mean(ret_tensors["uq"], axis=0)
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if self.num_models > 1:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy() * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if self.num_models > 1:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )
        if self.model_type in ["DipoleMACE", "EnergyDipoleMACE"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            if self.num_models > 1:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )
