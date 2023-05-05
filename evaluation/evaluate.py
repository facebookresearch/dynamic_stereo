# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import hydra
import numpy as np

import torch
from omegaconf import OmegaConf

from dynamic_stereo.evaluation.utils.utils import aggregate_and_print_results

import dynamic_stereo.datasets.dynamic_stereo_datasets as datasets

from dynamic_stereo.models.core.model_zoo import (
    get_all_model_default_configs,
    model_zoo,
)
from pytorch3d.implicitron.tools.config import get_default_args_field
from dynamic_stereo.evaluation.core.evaluator import Evaluator


@dataclass(eq=False)
class DefaultConfig:
    exp_dir: str = "./outputs"

    # one of [sintel, dynamicreplica, things]
    dataset_name: str = "dynamicreplica"

    sample_len: int = -1
    dstype: Optional[str] = None
    # clean, final
    MODEL: Dict[str, Any] = field(
        default_factory=lambda: get_all_model_default_configs()
    )
    EVALUATOR: Dict[str, Any] = get_default_args_field(Evaluator)

    seed: int = 42
    gpu_idx: int = 0

    visualize_interval: int = 0  # Use 0 for no visualization

    # Override hydra's working directory to current working dir,
    # also disable storing the .hydra logs:
    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},
            "output_subdir": None,
        }
    )


def run_eval(cfg: DefaultConfig):
    """
    Evaluates new view synthesis metrics of a specified model
    on a benchmark dataset.
    """
    # make the experiment directory
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # dump the exp cofig to the exp_dir
    cfg_file = os.path.join(cfg.exp_dir, "expconfig.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    evaluator = Evaluator(**cfg.EVALUATOR)

    model = model_zoo(**cfg.MODEL)
    model.cuda(0)
    evaluator.setup_visualization(cfg)

    if cfg.dataset_name == "dynamicreplica":
        test_dataloader = datasets.DynamicReplicaDataset(
            split="valid", sample_len=cfg.sample_len, only_first_n_samples=1
        )
    elif cfg.dataset_name == "sintel":
        test_dataloader = datasets.SequenceSintelStereo(dstype=cfg.dstype)
    elif cfg.dataset_name == "things":
        test_dataloader = datasets.SequenceSceneFlowDatasets(
            {},
            dstype=cfg.dstype,
            sample_len=cfg.sample_len,
            add_monkaa=False,
            add_driving=False,
            things_test=True,
        )
    elif cfg.dataset_name == "real":
        for real_sequence_name in ["teddy_static", "ignacio_waving", "nikita_reading"]:
            ds_path = f"./dynamic_replica_data/real/{real_sequence_name}"
            # seq_len_real = 20
            real_dataset = datasets.DynamicReplicaDataset(
                split="test",
                sample_len=cfg.sample_len,
                root=ds_path,
                only_first_n_samples=1,
            )

            evaluator.evaluate_sequence(
                model=model,
                test_dataloader=real_dataset,
                is_real_data=True,
                train_mode=False,
            )
        return

    print()

    evaluate_result = evaluator.evaluate_sequence(
        model,
        test_dataloader,
    )

    aggreegate_result = aggregate_and_print_results(evaluate_result)

    result_file = os.path.join(cfg.exp_dir, f"result_eval.json")

    print(f"Dumping eval results to {result_file}.")
    with open(result_file, "w") as f:
        json.dump(aggreegate_result, f)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


@hydra.main(config_path="./configs/", config_name="default_config_eval")
def evaluate(cfg: DefaultConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_idx)
    run_eval(cfg)


if __name__ == "__main__":
    evaluate()
