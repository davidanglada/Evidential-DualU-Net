#!/usr/bin/env python3
# test_dualunet_edl.py
#
# Test/evaluate DualUNet with Evidential heads (seg + centroid) on circles (or similar).
# Uses evaluate_test from the DualUNet engine (packs both p_seg and p_cent).

import argparse
import os
import os.path as osp
import sys
import numpy as np

import torch
import torch.nn as nn
import wandb

from dual_unet.utils.distributed import init_distributed_mode, get_rank, is_main_process
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config
from dual_unet.datasets import build_dataset, build_loader
from dual_unet.models import build_model
# ⬇️ DualUNet engine evaluate_test (the one that returns both seg/cent metrics)
from dual_unet.engine import evaluate_test


def _resolve_ckpt_path(cfg) -> str:
    """
    Resolve a checkpoint path in this order:
      1) cfg.experiment.ckpt_path (if provided)
      2) <output_dir>/<output_name>_best.pth
      3) <output_dir>/<output_name>.pth
      4) <output_dir>/<output_name>  (as-is, for backwards compatibility)
    """
    exp = cfg["experiment"]
    if "ckpt_path" in exp and exp["ckpt_path"]:
        return exp["ckpt_path"]

    base = osp.join(exp["output_dir"], exp["output_name"])
    candidates = [f"{base}_best.pth", f"{base}.pth", base]
    for p in candidates:
        if osp.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find a checkpoint. Tried: {', '.join(candidates)}. "
        "Provide experiment.ckpt_path in the config or ensure the file exists."
    )


def test(cfg):
    """
    Test/evaluation entry-point for the (Evidential) DualUNet (seg + centroid).
    """
    # 1) Distributed setup & device
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # 2) Optional wandb
    cfg['experiment']['wandb'] = cfg['experiment'].get('wandb', False)
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.init(
            project=cfg['experiment']['project'],
            name=cfg['experiment']['name'],
            config=cfg,
            group=cfg['experiment'].get('wandb_group', None)
        )

    # 3) Reproducibility
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # 4) Data
    test_dataset = build_dataset(cfg, split='test')
    test_loader  = build_loader(cfg, test_dataset, split='test')

    # 5) Model (+ a no-op "criterion" for interface symmetry)
    model = build_model(cfg).to(device)
    criterion = nn.Identity().to(device)  # engine.evaluate_test ignores it

    # 6) Load checkpoint
    ckpt_path = _resolve_ckpt_path(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    load_status = model.load_state_dict(state_dict, strict=True)
    try:
        missing = getattr(load_status, "missing_keys", [])
        unexpected = getattr(load_status, "unexpected_keys", [])
    except Exception:
        missing, unexpected = [], []
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"\tMissing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")

    # 7) DDP wrap (optional)
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])

    # 8) Evaluate (DualUNet → evaluator consumes p_seg & p_cent)
    eval_cfg = cfg.get('evaluation', {})
    thresholds = eval_cfg.get('thresholds', [0.5])                 # for centroid thresholding buckets (if used)
    max_pair_distance = eval_cfg.get('max_pair_distance', 12.0)    # pairing radius (pixels)
    th_centroid = eval_cfg.get('centroid_h', 0.15)                 # h-maxima / centroid peak threshold

    test_stats = evaluate_test(
        cfg=cfg,
        model=model,
        criterion=criterion,                 # kept for signature symmetry; not used
        data_loader=test_loader,
        device=device,
        thresholds=thresholds,
        max_pair_distance=max_pair_distance,
        output_sufix=cfg['experiment']['name'],
        train=False,
        th=th_centroid,
    )

    # The DualUNet engine returns a dict like {"f": {...}} → flatten with "test_" prefix
    if isinstance(test_stats, dict) and "f" in test_stats and isinstance(test_stats["f"], dict):
        stats = {f"test_{k}": v for k, v in test_stats["f"].items()}
    else:
        stats = {f"test_{k}": v for k, v in test_stats.items()}

    print(stats)

    if cfg['experiment']['wandb'] and is_main_process():
        wandb.log(stats)

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test/evaluate DualUNet (EDL seg + centroid).')
    parser.add_argument('--config-file', type=str, required=True, help='Path to config file.')
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="Override options like key1=value1 key2=value2")
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    # simple overrides with basic typing
    if args.opts:
        for opt in args.opts:
            k, v = opt.split('=')
            if v.lower() in ('true', 'false'):
                v = (v.lower() == 'true')
            else:
                try:
                    if '.' in v: v = float(v)
                    else: v = int(v)
                except ValueError:
                    pass
            d = cfg
            keys = k.split('.')
            for kk in keys[:-1]:
                d = d[kk]
            d[keys[-1]] = v

    test(cfg)
