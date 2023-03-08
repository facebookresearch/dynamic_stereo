# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import Configurable
from pytorch3d.implicitron.tools.vis_utils import get_visdom_connection


class BaseEvaluator(Configurable):
    """
    A class defining Base evaluator.
    """

    def setup_visualization(self, cfg: DictConfig) -> None:
        # Visualization
        self.visualize_interval = cfg.visualize_interval
        self.visdom_env = cfg.visdom_env
        self.exp_dir = cfg.exp_dir
        if self.visualize_interval > 0:
            self.viz = get_visdom_connection(cfg.visdom_server, cfg.visdom_port)
            self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
            self.visualize_dir = os.path.join(cfg.exp_dir, "visualisations")

    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        preprocess_result: None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "evaluate_sequence is not implemented in BaseEvaluator"
        )
