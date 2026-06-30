import argparse
import os.path as osp
import sys
import os
import numpy as np

import torch
import torch.nn as nn

from collections import OrderedDict

# Add project path if needed
sys.path.append('../dual_unet')

# Distributed / logging
from dual_unet.utils.distributed import init_distributed_mode, save_on_master, is_main_process, get_rank
from dual_unet.utils.misc import seed_everything
from dual_unet.utils.config import load_config

# Data, Model, Engine
from dual_unet.datasets import (
    build_dataset,
    build_loader,
    compute_class_weights_with_background,
    compute_class_weights_no_background
)
from dual_unet.models import build_model, load_state_dict
from dual_unet.engine import train_one_epoch, evaluate

# Losses
from dual_unet.models.losses import DualLoss_Evidential

import wandb

def effective_weights_from_old(w_old, beta=0.98):
    w_old = np.asarray(w_old, dtype=float)
    
    # 1) reconstruct relative frequencies
    rel_counts = 1.0 / w_old
    rel_counts = rel_counts / rel_counts.sum()

    # 2) compute effective numbers
    E = (1 - beta**rel_counts) / (1 - beta)

    # 3) invert + normalize to get new weights
    w_new = (1.0 / E)
    w_new = w_new / w_new.sum()

    return w_new



def train(cfg: dict):
    """
    Main training loop for Dual U-Net with Evidential heads:
      - Segmentation head: Dirichlet (K classes)
      - Centroid head:     NIG evidential regression on Gaussian centroid map
    """
    # Step 1: Initialize distributed
    torch.backends.cudnn.benchmark = False
    init_distributed_mode(cfg)
    device = torch.device(f"cuda:{cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    # Step 2: Initialize wandb if needed
    cfg['experiment']['wandb'] = bool(cfg['experiment'].get('wandb', False))
    if cfg['experiment']['wandb'] and is_main_process():
        wandb.init(
            project=cfg['experiment']['project'],
            name=cfg['experiment']['name'],
            config=cfg
        )

    # Step 3: Seed
    seed = cfg['experiment']['seed'] + get_rank()
    seed_everything(seed)

    # Step 4: Build train & val datasets/loaders
    train_dataset = build_dataset(cfg, split='train')
    val_dataset   = build_dataset(cfg, split='val')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size:   {len(val_dataset)}")

    train_loader = build_loader(cfg, train_dataset, split='train')
    val_loader   = build_loader(cfg, val_dataset,   split='val')
    print("Data loaders created.")

    # Step 5: Build model
    model = build_model(cfg)
    print("Model built.")

    # Step 6: Compute or load class weights (for seg data term & optional Dice)
    if cfg['dataset']['train']['name'] == 'cell':
        ce_weights = torch.tensor([1, 1]).to(device)  # Dummy weights for 1-class cell dataset
    else:
        ce_weights_path = cfg['training'].get('ce_weights', 'ce_weights.npy')
        if not osp.exists(ce_weights_path):
            ce_weights = compute_class_weights_with_background(
                train_dataset,
                cfg['dataset']['train']['num_classes'],
                background_importance_factor=10
            ).to(device)
            np.save(ce_weights_path, ce_weights.cpu().numpy())
        else:
            ce_weights = torch.tensor(np.load(ce_weights_path)).to(device)
    
    
    ce_weights = torch.tensor(effective_weights_from_old(ce_weights.cpu().numpy(), beta=0.98)).to(device)
    # ----------------------------------------------------
    # LOSS SELECTION  (based on cfg['training'])
    # ----------------------------------------------------
    print("[INFO] Building DualLoss_Evidential from cfg['training'] only")

    # Common knobs
    num_classes   = int(cfg["dataset"]["train"]["num_classes"] + 1)

    tr            = cfg.get("training", {})
    seg_mode      = str(tr.get("seg_mode", "edl_mse"))

    weight_seg    = float(tr.get("weight_seg", 1.0))
    weight_cent   = float(tr.get("weight_cent", 1.0))
    weight_dice   = float(tr.get("weight_dice", 0.0))
    dice_gamma    = float(tr.get("dice_gamma", 1.0))

    lbl_smooth    = float(tr.get("label_smoothing", 0.0))
    ignore_index  = tr.get("ignore_index", None)

    # Class weights
    cw_list = tr.get("class_weights", None)
    class_weights = (
        torch.tensor(cw_list, device=device, dtype=torch.float32)
        if cw_list is not None else None
    )

    # Centroid positives weighting (legacy, for Dirichlet; ignored in NIG but kept for API)
    pos_weight_cent = float(tr.get("pos_weight_cent", 1.0))

    # KL controls (seg)
    kl_max          = float(tr.get("kl_max", 1e-3))
    kl_schedule     = str(tr.get("kl_schedule", "ramp"))          # {"constant","ramp"}
    kl_ramp_epochs  = int(tr.get("kl_ramp_epochs", 40))
    kl_variant_seg  = str(tr.get("kl_variant_seg", "vanilla"))     # {"sensoy","vanilla"}
    kl_apply_seg    = str(tr.get("kl_apply_seg", "misclassified")) # {"all","misclassified"}
    kl_boost_seg    = float(tr.get("kl_boost_seg", 1.0))
    kl_entropy_gate_seg = bool(tr.get("kl_entropy_gate_seg", False))
    kl_entropy_thr_seg  = float(tr.get("kl_entropy_thr_seg", 0.65))

    # UWG / BSCE-GRA controls (seg head)
    uwg_enable     = bool(tr.get("uwg_enable", False))
    uwg_style      = str(tr.get("uwg_style", "gate"))      # {"gate","brier"}; "brier" ~ BSCE-GRA
    uwg_proxy      = str(tr.get("uwg_proxy", "mse"))       # {"entropy","mse"} (for "gate")
    uwg_gamma      = float(tr.get("uwg_gamma", 10.0))      # gate steepness or brier exponent
    uwg_tau_mode   = str(tr.get("uwg_tau_mode", "median")) # {"median","fixed"}
    uwg_tau        = float(tr.get("uwg_tau", 0.5))
    uwg_norm_mean1 = bool(tr.get("uwg_norm_mean1", True))
    eps_brier      = float(tr.get("eps_brier", 1e-8))

    # Build the unified criterion
    criterion = DualLoss_Evidential(
        num_classes=num_classes,
        class_weights=ce_weights,

        # data terms
        seg_mode=seg_mode,
        weight_seg=weight_seg,
        weight_cent=weight_cent,
        weight_dice=weight_dice,
        dice_gamma=dice_gamma,
        seg_label_smoothing=lbl_smooth,
        ignore_index=ignore_index,
        pos_weight_cent=pos_weight_cent,          # legacy; ignored by NIG

        # KL (seg)
        kl_max=kl_max,
        kl_schedule=kl_schedule,
        kl_ramp_epochs=kl_ramp_epochs,
        kl_variant_seg=kl_variant_seg,
        kl_apply_seg=kl_apply_seg,
        kl_boost_seg=kl_boost_seg,
        kl_entropy_gate_seg=kl_entropy_gate_seg,
        kl_entropy_thr_seg=kl_entropy_thr_seg,

        # UWG / BSCE-GRA (seg)
        uwg_enable=uwg_enable,
        uwg_style=uwg_style,
        uwg_proxy=uwg_proxy,
        uwg_gamma=uwg_gamma,
        uwg_tau_mode=uwg_tau_mode,
        uwg_tau=uwg_tau,
        uwg_norm_mean1=uwg_norm_mean1,
        eps_brier=eps_brier,
    )

    print(
        "[INFO] Loss configured:",
        f"seg_mode={seg_mode}, dice_w={weight_dice}, "
        f"kl_max={kl_max}, kl_variant_seg={kl_variant_seg}/{kl_apply_seg}, "
    )

    # Move model & criterion to device
    model.to(device)
    criterion.to(device)
    print("Model and criterion prepared.")

    # Step 8: Possibly load an existing checkpoint
    load_state_dict(cfg, model)

    # Step 9: Setup optimizer
    base_lr = cfg['optimizer']['lr_base']
    lr_auto_scale = cfg['optimizer'].get('lr_auto_scale', False)
    lr_scale = 1.0
    if lr_auto_scale:
        # Simple batch-size based scaling (relative to a reference BS)
        ref_bs = cfg['optimizer'].get('base_batch_size', 16)
        actual_bs = cfg['loader']['train']['batch_size'] * max(1, cfg.get('world_size', 1))
        lr_scale = float(actual_bs) / float(ref_bs)
        print(f"Auto-scaling LR by factor {lr_scale:.3f} (actual_bs={actual_bs}, ref_bs={ref_bs})")

    # Two param groups: (a) everything except centroid head, (b) centroid head (slightly lower LR)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "count_head" not in n],
            "lr": base_lr * lr_scale,
        },
        {
            # NIG centroid head: gamma, nu, alpha, beta maps
            "params": list(model.count_head.parameters()),
            "lr": base_lr * lr_scale * cfg['optimizer'].get('centroid_head_lr_mult', 0.5)
        }
    ]
    optimizer = torch.optim.Adam(
        param_dicts,
        lr=base_lr * lr_scale,
        weight_decay=cfg['optimizer']['weight_decay']
    )

    # Step 10: Setup LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg['optimizer']['lr_drop_steps'],
        gamma=cfg['optimizer']['lr_drop_factor']
    )

    # If distributed
    if cfg['distributed']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['gpu']])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Possibly resume from checkpoint
    curr_epoch = 1
    if cfg['experiment'].get('resume', False):
        output_dir = cfg['experiment']['output_dir']
        output_name = cfg['experiment']['output_name']
        ckpt_path = osp.join(output_dir, output_name)
        if osp.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model_without_ddp.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            curr_epoch = ckpt.get('epoch', 1)
            print(f"Resumed from checkpoint {ckpt_path} at epoch {curr_epoch}.")

    max_epochs    = cfg['optimizer']['epochs']
    eval_interval = cfg['evaluation'].get('interval', 10)
    save_interval = 50
    th_centroid   = cfg['evaluation'].get('th_centroid', 0.15)
    best_val = None
    best_ckpt_path = None

    for epoch in range(curr_epoch, max_epochs + 1):
        print(f"Starting epoch {epoch} / {max_epochs}...")
        if cfg['distributed']:
            # set epoch in sampler for correct shuffling
            train_loader.sampler.set_epoch(epoch)

        # Training
        train_stats = train_one_epoch(
            cfg, model, criterion, train_loader, optimizer, device, epoch
        )

        lr_scheduler.step()

        # Evaluate if needed
        val_stats = {}
        if epoch == 1 or epoch == max_epochs or (epoch % eval_interval == 0):
            print("Evaluating on validation set...")
            val_stats = evaluate(
                model, criterion, val_loader, device,
                max_pair_distance=cfg['evaluation']['max_pair_distance'],
                th=th_centroid
            )
            print(f"Epoch {epoch} Validation Stats: {val_stats}")

            # Track a primary metric (use validation loss; swap to dice if preferred)
            primary = val_stats.get('loss', None)
            if primary is not None:
                best_val = primary
                # Save "best" checkpoint periodically
                output_dir = cfg['experiment'].get('output_dir', None)
                output_name = f"{cfg['experiment'].get('output_name', 'model_best.pth')}_{epoch}.pth"
                if output_dir and output_name and (epoch % save_interval == 0):
                    ckpt = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch
                    }
                    best_ckpt_path = osp.join(output_dir, f"{output_name}")
                    save_on_master(ckpt, best_ckpt_path)
                    print(f"Best checkpoint (loss={best_val:.4f}) saved to {best_ckpt_path}")

            # Optional periodic snapshot
            if cfg['experiment'].get('save_every_eval', False):
                output_dir = cfg['experiment'].get('output_dir', None)
                output_name = cfg['experiment'].get('output_name', "snapshot")
                if output_dir and output_name:
                    ckpt_path = osp.join(output_dir, f"{output_name}_epoch_{epoch}.pth")
                    ckpt = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch
                    }
                    save_on_master(ckpt, ckpt_path)
                    print(f"Snapshot checkpoint saved to {ckpt_path}")

        # Log to wandb if main process
        if cfg['experiment']['wandb'] and is_main_process():
            log = {}

            def _pref(d, pfx):
                return {f"{pfx}/{k}": (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in d.items()}

            log.update(_pref(train_stats, "train"))
            if val_stats:
                log.update(_pref(val_stats, "val"))

            if "stats" in train_stats and isinstance(train_stats["stats"], dict):
                log.update(_pref(train_stats["stats"], "train"))
            if val_stats and "stats" in val_stats and isinstance(val_stats["stats"], dict):
                log.update(_pref(val_stats["stats"], "val"))

            wanted = [
                "loss","loss_seg","loss_cent","loss_dice","loss_kl","loss_kl_seg","loss_kl_cent",
                "mean_S_seg","mean_max_p_seg","misclassified_frac",
                "mean_S_cent","mean_max_p_cent","mean_p_centroid","kl_weight",
                "lr","grad_norm"
            ]
            for k in wanted:
                if k in train_stats: log[f"train/{k}"] = float(train_stats[k])
                if val_stats and k in val_stats: log[f"val/{k}"] = float(val_stats[k])

            wandb.log(log)

    # End training
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DualUNet (Evidential)")
    parser.add_argument('--config-file', type=str, default=None, help='Path to config file')
    parser.add_argument("--opts", help="Override config options in key=value format",
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config_file is not None, "Please provide a --config-file path."

    cfg = load_config(args.config_file)

    # Possibly override cfg options from cmd line
    if args.opts is not None:
        for opt in args.opts:
            k, v = opt.split('=')
            # Simple type inference
            if v.isdigit():
                v = int(v)
            elif v.replace('.', '', 1).isdigit():
                v = float(v)
            elif v.lower() in ['true', 'false']:
                v = (v.lower() == 'true')
            # Nested keys
            keys = k.split('.')
            d = cfg
            for key_part in keys[:-1]:
                d = d[key_part]
            d[keys[-1]] = v

    train(cfg)
