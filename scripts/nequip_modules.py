import os
import torch
import numpy as np
from typing import Optional
from torch.distributed import is_initialized

from nequip.train.ema import EMALightningModule
from nequip.data import AtomicDataDict, register_fields

ATOM_ID_KEY = "global_atom_id"
register_fields(node_fields=[ATOM_ID_KEY])

class LTAULightningModule(EMALightningModule):
    def __init__(self, *args, error_save_dir: Optional[str] = "force_errors", **kwargs):
        super().__init__(*args, **kwargs)
        self.error_save_dir = error_save_dir
        os.makedirs(self.error_save_dir, exist_ok=True)
        self._error_buffer = None
        self._epoch_force_errors = []  # used in fallback mode

    def setup(self, stage=None):
        super().setup(stage)
        if stage != "fit" or self._error_buffer is not None:
            return

        train_ds = self.trainer.datamodule.train_dataset
        datasets = train_ds if isinstance(train_ds, list) else [train_ds]

        all_atom_ids = []
        for ds_list in datasets:
            for ds in ds_list:
                # if not hasattr(ds, "data_list"):
                #     continue
                for data in ds:#.data_list:
                    ids = data.get(ATOM_ID_KEY, None)
                    if ids is not None:
                        all_atom_ids.extend(torch.as_tensor(ids).tolist())

        if not all_atom_ids:
            raise RuntimeError("No atom IDs found in dataset during setup().")

        max_id = max(all_atom_ids)
        self._error_buffer = torch.full((max_id + 1,), float("nan"))

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        target = self.process_target(batch, batch_idx, dataloader_idx)
        output = self(batch)

        if self.train_metrics is not None:
            with torch.no_grad():
                train_metric_dict = self.train_metrics(
                    output, target, prefix=f"train_metric_step{self.logging_delimiter}"
                )
            self.log_dict(train_metric_dict)

        pred_forces = output[AtomicDataDict.FORCE_KEY]
        true_forces = batch[AtomicDataDict.FORCE_KEY]
        per_atom_error = torch.norm(pred_forces - true_forces, dim=-1)

        atom_ids = batch.get(
            ATOM_ID_KEY,
            torch.arange(per_atom_error.shape[0], device=per_atom_error.device)
        ).long()

        # Write into fixed-size error buffer
        if self._error_buffer is not None:
            atom_ids = atom_ids.detach().cpu()
            errors = per_atom_error.detach().cpu()
            for i, aid in enumerate(atom_ids):
                self._error_buffer[aid] = errors[i]
        else:
            # Fallback if error_buffer wasn't set up
            per_atom_data = torch.stack(
                [atom_ids.detach().to(torch.float32), per_atom_error.detach()],
                dim=1
            )
            self._epoch_force_errors.append(per_atom_data.cpu())

        loss_dict = self.loss(
            output, target, prefix=f"train_loss_step{self.logging_delimiter}"
        )
        self.log_dict(loss_dict)

        loss = (
            loss_dict[f"train_loss_step{self.logging_delimiter}weighted_sum"]
            * self.world_size
        )
        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        if self._error_buffer is not None:
            local_errors = self._error_buffer.to(self.device)
            if is_initialized():
                gathered = [
                    torch.empty_like(local_errors) for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(gathered, local_errors)
                if self.trainer.is_global_zero:
                    all_errors = torch.stack(gathered, dim=0)
                    # Use the first non-NaN value per atom
                    # final_errors = torch.nanmin(all_errors, dim=0).values
                    final_errors = nanmin_stack(all_errors)
                else:
                    final_errors = None
            else:
                final_errors = local_errors

            if self.trainer.is_global_zero and final_errors is not None:
                epoch = self.trainer.current_epoch
                file_path = os.path.join(self.error_save_dir, f"train_force_errors_epoch_{epoch}.npy")
                np.save(file_path, final_errors.cpu().numpy())

            self._error_buffer[:] = float("nan")

        elif self._epoch_force_errors:
            # fallback mode: gather individual entries
            local_errors = torch.cat(self._epoch_force_errors, dim=0).to(self.device)

            if is_initialized():
                gathered = [torch.zeros_like(local_errors) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered, local_errors)
                if self.trainer.is_global_zero:
                    all_errors = torch.cat(gathered, dim=0)
                else:
                    all_errors = None
            else:
                all_errors = local_errors

            if self.trainer.is_global_zero and all_errors is not None:
                atom_ids = all_errors[:, 0].cpu().numpy()
                assert len(np.unique(atom_ids)) == len(atom_ids), (
                    f"Duplicate atom_ids found: {len(atom_ids)} total vs {len(np.unique(atom_ids))} unique"
                )

                sorted_errors = all_errors[all_errors[:, 0].argsort()]
                epoch = self.trainer.current_epoch
                file_path = os.path.join(self.error_save_dir, f"train_force_errors_epoch_{epoch}.npy")
                np.save(file_path, sorted_errors.cpu().numpy())

            self._epoch_force_errors = []

def nanmin_stack(tensors: torch.Tensor) -> torch.Tensor:
    """Compute nanmin across 0th dimension of a stacked tensor (N, M)."""
    valid_mask = ~torch.isnan(tensors)
    masked = torch.where(valid_mask, tensors, float("inf"))
    return torch.min(masked, dim=0).values

import torch
from typing import List, Optional, Union, Dict, Callable

from nequip.data import AtomicDataDict
from nequip.data.dataset import ASEDataset
from nequip.data.datamodule import NequIPDataModule
from omegaconf import ListConfig, DictConfig, OmegaConf


class ASEDatasetWithGlobalIDs(ASEDataset):
    """ASEDataset subclass that assigns globally unique atom IDs on construction."""
    _global_counter = 0  # Class-level counter shared across all datasets

    def __init__(
        self,
        file_path: str,
        transforms: List[Callable] = [],
        ase_args: Dict[str, any] = {},
        include_keys: Optional[List[str]] = [],
        exclude_keys: Optional[List[str]] = [],
        key_mapping: Optional[Dict[str, str]] = {},
    ):
        super().__init__(
            file_path=file_path,
            transforms=transforms,
            ase_args=ase_args,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            key_mapping=key_mapping,
        )

        # Assign unique global atom IDs once at construction
        counter = 0
        for data in self.data_list:
            num_atoms = data[AtomicDataDict.POSITIONS_KEY].shape[0]
            data[ATOM_ID_KEY] = torch.arange(
                counter,
                counter + num_atoms,
                # ASEDatasetWithGlobalIDs._global_counter,
                # ASEDatasetWithGlobalIDs._global_counter + num_atoms,
                dtype=torch.long,
            )
            counter += num_atoms
            # ASEDatasetWithGlobalIDs._global_counter += num_atoms

        print(f'{counter=}')

        # # Reset in case it gets re-read by other process
        # ASEDatasetWithGlobalIDs._global_counter = 0

class ASEDataModuleWithID(NequIPDataModule):
    """ASEDataModule that uses ASEDatasetWithGlobalIDs to assign atom IDs only
    once."""
    
    def __init__(
        self,
        seed: int,
        # file paths
        train_file_path: Optional[Union[str, List[str]]] = [],
        val_file_path: Optional[Union[str, List[str]]] = [],
        test_file_path: Optional[Union[str, List[str]]] = [],
        predict_file_path: Optional[Union[str, List[str]]] = [],
        split_dataset: Optional[Union[Dict, List[Dict]]] = [],
        # data transforms
        transforms: List[Callable] = [],
        # ase params
        ase_args: dict = {},
        include_keys: Optional[List[str]] = [],
        exclude_keys: Optional[List[str]] = [],
        key_mapping: Optional[Dict[str, str]] = {},
        **kwargs,
    ):

        # == first convert all dataset paths to lists if not already lists ==
        dataset_paths = []
        for paths in [
            train_file_path,
            val_file_path,
            test_file_path,
            predict_file_path,
            split_dataset,
        ]:
            # convert to primitives as later logic is based on types
            if isinstance(paths, ListConfig) or isinstance(paths, DictConfig):
                paths = OmegaConf.to_container(paths, resolve=True)
            assert (
                isinstance(paths, list)
                or isinstance(paths, str)
                or isinstance(paths, dict)
            )
            if not isinstance(paths, list):
                # convert str -> List[str]
                dataset_paths.append([paths])
            else:
                dataset_paths.append(paths)

        # == assemble config template ==
        dataset_config_template = {
            # "_target_": "nequip.data.dataset.ASEDataset",
            "_target_": "ltau_nequip_modules.ASEDatasetWithGlobalIDs",
            "transforms": transforms,
            "ase_args": ase_args,
            "include_keys": include_keys,
            "exclude_keys": exclude_keys,
            "key_mapping": key_mapping,
        }

        # == populate train, val, test predict, split datasets ==
        dataset_configs = [[], [], [], []]
        for config, paths in zip(dataset_configs, dataset_paths[:-1]):
            for path in paths:
                dataset_config = dataset_config_template.copy()
                dataset_config.update({"file_path": path})
                config.append(dataset_config)

        # == populate split dataset ==
        split_config = []
        for path_and_splits in dataset_paths[-1]:
            assert (
                "file_path" in path_and_splits
            ), "`file_path` key must be present in each dict of `split_dataset`"
            dataset_config = dataset_config_template.copy()
            file_path = path_and_splits.pop("file_path")
            dataset_config.update({"file_path": file_path})
            path_and_splits.update(
                {"dataset": dataset_config}
            )  # now actually dataset_and_splits
            split_config.append(path_and_splits)

        super().__init__(
            seed=seed,
            train_dataset=dataset_configs[0],
            val_dataset=dataset_configs[1],
            test_dataset=dataset_configs[2],
            predict_dataset=dataset_configs[3],
            split_dataset=split_config,
            **kwargs,
        )
