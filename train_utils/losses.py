# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt().unsqueeze(1)

    if len(valid.shape) != len(flow_gt.shape):
        valid = valid.unsqueeze(1)

    valid = (valid >= 0.5) & (mag < max_flow)

    if valid.shape != flow_gt.shape:
        valid = torch.cat([valid, valid], dim=1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )

        if n_predictions == 1:
            i_weight = 1
        else:
            # We adjust the loss_gamma so it is consistent for any number of iterations
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        flow_pred = flow_preds[i].clone()
        if valid.shape[1] == 1 and flow_preds[i].shape[1] == 2:
            flow_pred = flow_pred[:, :1]

        i_loss = (flow_pred - flow_gt).abs()

        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            flow_gt.shape,
            flow_pred.shape,
        ]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    valid = valid[:, 0]
    epe = epe.view(-1)
    epe = epe[valid.reshape(epe.shape)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics
