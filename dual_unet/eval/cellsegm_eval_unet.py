#!/usr/bin/env python3
# unet_eval_watershed_circles.py (revised, robust label space)

import datetime
import itertools
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchmetrics.functional as TMF
import cv2
import matplotlib.pyplot as plt
import os
import os.path as osp
import torchvision.transforms.v2 as v2

from scipy.ndimage import distance_transform_edt, label as cc_label
from skimage.morphology import extrema
from skimage.segmentation import watershed, find_boundaries

# Optional PQ helpers
try:
    from ..utils.distributed import all_gather
    from .pq import compute_bPQ_and_mPQ, remap_label_and_class_map
except Exception:
    all_gather = None
    compute_bPQ_and_mPQ = None
    remap_label_and_class_map = None


# ---------------------------
# Generic helpers
# ---------------------------

def get_dice_1(true_map: np.ndarray, pred_map: np.ndarray) -> float:
    t = (true_map > 0).astype(np.uint8)
    p = (pred_map > 0).astype(np.uint8)
    inter = (t & p).sum()
    denom = t.sum() + p.sum()
    return 2.0 * inter / denom if denom > 0 else 0.0

def ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            acc_bin  = correct[m].mean()
            conf_bin = conf[m].mean()
            ece     += m.mean() * abs(acc_bin - conf_bin)
    return float(ece)

def distance_peaks_markers(fg_mask: np.ndarray, h: float = 0.1):
    dist_map = distance_transform_edt(fg_mask.astype(np.uint8))
    if dist_map.max() > 0:
        dist_norm = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min() + 1e-12)
    else:
        dist_norm = dist_map
    hmax = extrema.h_maxima(dist_norm, h=h)
    markers, _ = cc_label(hmax.astype(np.uint8))
    return markers, dist_map

# ---------- IoU pairing (fallback) ----------
def _instance_iou_matrix(gt_map: np.ndarray, pred_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_ids = np.unique(gt_map); gt_ids = gt_ids[gt_ids != 0]
    pr_ids = np.unique(pred_map); pr_ids = pr_ids[pr_ids != 0]
    n, m = len(gt_ids), len(pr_ids)
    iou = np.zeros((n, m), dtype=np.float32)
    if n == 0 or m == 0:
        return iou, gt_ids, pr_ids
    for i, g in enumerate(gt_ids):
        gmask = (gt_map == g)
        gsum = gmask.sum()
        if gsum == 0:
            continue
        for j, p in enumerate(pr_ids):
            pmask = (pred_map == p)
            inter = (gmask & pmask).sum()
            if inter == 0:
                continue
            union = gsum + pmask.sum() - inter
            iou[i, j] = inter / max(1, union)
    return iou, gt_ids, pr_ids

def _hungarian_match_iou(iou: np.ndarray, thr: float = 0.5) -> np.ndarray:
    if iou.size == 0:
        return np.empty((0, 2), dtype=int)
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(1.0 - iou)  # maximize IoU
    keep = iou[rows, cols] >= thr
    return np.stack([rows[keep], cols[keep]], axis=1)

# ---------- Centroid pairing (preferred) ----------
def pair_coordinates(setA: np.ndarray, setB: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if setA.size == 0 or setB.size == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(setA.shape[0], dtype=int),
                np.arange(setB.shape[0], dtype=int))
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    D = cdist(setA, setB, metric="euclidean")
    rows, cols = linear_sum_assignment(D)
    dsel = D[rows, cols]
    keep = dsel <= radius
    paired = np.stack([rows[keep], cols[keep]], axis=1)
    unpairedA = np.delete(np.arange(setA.shape[0]), rows[keep])
    unpairedB = np.delete(np.arange(setB.shape[0]), cols[keep])
    return paired, unpairedA, unpairedB

# ---------- viz helpers ----------
def _denormalize(
    image: Union[torch.Tensor, np.ndarray],
    mean: List[float] = [0.485, 0.456, 0.406],
    std:  List[float] = [0.229, 0.224, 0.225]
):
    if isinstance(image, torch.Tensor):
        if image.ndimension() == 3:
            mean = torch.tensor(mean).view(-1, 1, 1)
            std  = torch.tensor(std).view(-1, 1, 1)
        else:
            mean = torch.tensor(mean).view(1, 1, -1)
            std  = torch.tensor(std).view(1, 1, -1)
        return (image * std) + mean
    else:
        mean_arr = np.array(mean).reshape(-1, 1, 1)
        std_arr  = np.array(std).reshape(-1, 1, 1)
        return (image * std_arr) + mean_arr

def _image_from_norm(img: np.ndarray) -> torch.Tensor:
    img = _denormalize(img)
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    return transforms(img)


# ---------------------------
# Base Metric Buffer
# ---------------------------

class BaseMetricBuffer:
    def __init__(self):
        self.preds: List[Dict[str, torch.Tensor]] = []
        self.targets: List[Dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self.preds = []
        self.targets = []

    def update(self, preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
        self.preds.extend(preds)
        self.targets.extend(targets)

    def synchronize(self) -> None:
        if all_gather is None or (not dist.is_available()) or (not dist.is_initialized()):
            return
        dist.barrier()
        all_preds = all_gather(self.preds)
        all_tgts  = all_gather(self.targets)
        self.preds = list(itertools.chain(*all_preds))
        self.targets = list(itertools.chain(*all_tgts))


# ---------------------------
# UNet Evaluator
# ---------------------------

class UNetEvaluationMetric(BaseMetricBuffer):
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        eval_mode: bool = True,
        h: float = 0.1,
        output_suffix: Optional[str] = None,
        dataset_tag: str = "circles",
        pair_mode: str = "centroid",
        pair_radius: float = 12.0,
        iou_thr: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None else [str(i) for i in range(num_classes)]
        self.eval_mode = eval_mode
        self.h = h
        self.output_suffix = output_suffix if output_suffix is not None \
            else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.dataset_tag = dataset_tag
        self.pair_mode = pair_mode
        self.pair_radius = float(pair_radius)
        self.iou_thr = float(iou_thr)

    # sanitize labels to [0..K-1]
    @staticmethod
    def _sanitize_labels_to_K(arr: np.ndarray, K: int) -> np.ndarray:
        a = np.asarray(arr).astype(int, copy=False)
        a[(a < 0) | (a >= K)] = 0
        return a

    # convert GT (possibly one-hot/prob) to label map [H,W]
    @staticmethod
    def _to_label_map(gt: np.ndarray, K: int) -> np.ndarray:
        if gt.ndim == 3:
            # assume [K,H,W] → labels
            gt_lab = gt.argmax(axis=0)
        else:
            gt_lab = gt
        return UNetEvaluationMetric._sanitize_labels_to_K(gt_lab.astype(np.int64, copy=False), K)

    def compute(self) -> Dict[str, float]:
        self.synchronize()

        # Pull arrays
        images = [p["image"].detach().cpu().numpy() for p in self.preds]
        p_probs = [(p["p_hat"] if "p_hat" in p else p["segmentation_mask"]).detach().cpu().numpy()
                   for p in self.preds]
        t_seg   = [t["segmentation_mask"].detach().cpu().numpy() for t in self.targets]
        names   = [t["file_name"] for t in self.targets]

        if len(p_probs) == 0:
            return {}
        # Use K from predictions (robust even if some GTs are one-hot)
        K_pred = p_probs[0].shape[0]
        K = K_pred
        if (self.class_names is None) or (len(self.class_names) < K):
            self.class_names = [str(i) for i in range(K)]

        # Optional instance GT
        t_inst_masks = None
        if self.eval_mode and ("mask" in self.targets[0]):
            t_inst_masks = [t["mask"].detach().cpu().numpy() for t in self.targets]

        # Optional GT boxes+labels (for centroid pairing)
        has_boxes = ("boxes" in self.targets[0]) and ("labels" in self.targets[0])
        if has_boxes:
            t_boxes = [t["boxes"].detach().cpu().numpy() for t in self.targets]
            t_labels= [t["labels"].detach().cpu().numpy() for t in self.targets]

        # Clean heavy tensors
        for p in self.preds:
            for k in ["image", "p_hat", "segmentation_mask"]:
                if k in p: del p[k]
        for t in self.targets:
            for k in ["segmentation_mask", "mask", "boxes", "labels"]:
                if k in t: del t[k]
        torch.cuda.empty_cache()

        # Aggregates
        dice_scores, ece_conf, ece_corr, hn_dice_scores = [], [], [], []

        gt_inst_map, pred_inst_map, gt_class_map, pred_class_map = [], [], [], []

        det_TP = det_FP = det_FN = 0
        per_class_TP = np.zeros(K, dtype=np.int64)
        per_class_FP = np.zeros(K, dtype=np.int64)
        per_class_FN = np.zeros(K, dtype=np.int64)

        MAX_VIS = 10

        # Loop
        for i in range(len(p_probs)):
            probs = p_probs[i]      # [K,H,W]
            gt    = t_seg[i]        # [H,W] or [K,H,W]
            name  = names[i]
            img   = images[i]
            _, H, W = probs.shape

            # Coerce GT to label map and sanitize
            gt_lab = self._to_label_map(gt, K)

            # --- Pixel dice ---
            pred_labels = probs.argmax(axis=0)
            mean_dice = TMF.dice(torch.tensor(pred_labels),
                                 torch.tensor(gt_lab).int()).item()
            dice_scores.append(mean_dice)

            # --- ECE from max prob ---
            conf = probs.max(axis=0).ravel()
            corr = (pred_labels.ravel() == gt_lab.ravel()).astype(np.float32)
            ece_conf.append(conf); ece_corr.append(corr)

            # --- Watershed ---
            fg_mask = (pred_labels > 0).astype(np.uint8)
            markers, dist_map = distance_peaks_markers(fg_mask, h=self.h)
            ws = watershed(-dist_map, markers, mask=fg_mask, compactness=1)
            contours = np.invert(find_boundaries(ws, mode='outer', background=0))
            ws_clean = ws * contours

            # Label regions & majority class per instance
            labeled_ws, _ = cc_label(ws_clean > 0)  # instance ids 1..M
            pred_watershed_labels = np.zeros_like(pred_labels, dtype=np.int32)
            pred_centroids_yx, pred_cls = [], []
            for rid in np.unique(labeled_ws):
                if rid == 0: continue
                region = (labeled_ws == rid)
                maj = np.bincount(pred_labels[region]).argmax()
                pred_watershed_labels[region] = maj
                coords = np.argwhere(region)
                cy, cx = coords.mean(axis=0)
                pred_centroids_yx.append((float(cy), float(cx)))
                pred_cls.append(int(maj))
            pred_centroids_yx = np.array(pred_centroids_yx, dtype=np.float32)
            pred_cls = self._sanitize_labels_to_K(np.array(pred_cls, dtype=int), K)

            # --- Instance GT (optional) ---
            has_gt_masks = self.eval_mode and (t_inst_masks is not None) and (t_inst_masks[i] is not None)
            if has_gt_masks:
                gt_inst = np.zeros_like(labeled_ws, dtype=np.int32)
                inst_masks = t_inst_masks[i]  # (#inst,H,W)
                for k_, m_ in enumerate(inst_masks):
                    gt_inst[m_ > 0] = k_ + 1
                hn_dice = get_dice_1(gt_inst, labeled_ws)
                hn_dice_scores.append(hn_dice)

            # --- Detection pairing ---
            did_centroid = False
            if has_boxes and (self.pair_mode in {"centroid", "auto"}):
                boxes_i = t_boxes[i]
                labels_i = t_labels[i]
                if boxes_i.size > 0:
                    gt_centroids_yx = []
                    gt_cls = []
                    for b, lab in zip(boxes_i, labels_i):
                        x0, y0, w, h = b
                        cx = x0 + w // 2
                        cy = y0 + h // 2
                        gt_centroids_yx.append((float(cy), float(cx)))
                        gt_cls.append(int(lab))
                    gt_centroids_yx = np.array(gt_centroids_yx, dtype=np.float32)
                    gt_cls = self._sanitize_labels_to_K(np.array(gt_cls, dtype=int), K)

                    pairs, un_g, un_p = pair_coordinates(gt_centroids_yx, pred_centroids_yx, radius=self.pair_radius)
                    TP = pairs.shape[0]
                    FP = (0 if pred_centroids_yx.size == 0 else pred_centroids_yx.shape[0]) - TP
                    FN = (0 if gt_centroids_yx.size == 0 else gt_centroids_yx.shape[0]) - TP
                    det_TP += TP; det_FP += FP; det_FN += FN

                    for r, c in pairs:
                        gc = int(gt_cls[r]); pc = int(pred_cls[c])
                        if gc != 0:
                            if pc == gc: per_class_TP[gc] += 1
                            else:
                                per_class_FN[gc] += 1
                                if pc != 0: per_class_FP[pc] += 1

                    matched_pr = set(pairs[:, 1].tolist()) if TP > 0 else set()
                    for j in range(0 if pred_centroids_yx.size == 0 else pred_centroids_yx.shape[0]):
                        if j not in matched_pr:
                            c = int(pred_cls[j])
                            if c != 0: per_class_FP[c] += 1

                    matched_gt = set(pairs[:, 0].tolist()) if TP > 0 else set()
                    for j in range(0 if gt_centroids_yx.size == 0 else gt_centroids_yx.shape[0]):
                        if j not in matched_gt:
                            c = int(gt_cls[j])
                            if c != 0: per_class_FN[c] += 1

                    did_centroid = True

            if (not did_centroid) and has_gt_masks and (self.pair_mode in {"iou", "auto"}):
                iou, map_gt_ids, map_pr_ids = _instance_iou_matrix(gt_inst, labeled_ws)
                pairs = _hungarian_match_iou(iou, thr=self.iou_thr)
                TP = pairs.shape[0]
                pr_ids = np.unique(labeled_ws); pr_ids = pr_ids[pr_ids != 0]
                gt_ids = np.unique(gt_inst);    gt_ids = gt_ids[gt_ids != 0]
                FP = len(pr_ids) - TP
                FN = len(gt_ids) - TP
                det_TP += TP; det_FP += FP; det_FN += FN

                # class dicts via majority label (USE gt_lab instead of raw gt)
                pred_class_dict = {}
                for rid in pr_ids:
                    region = (labeled_ws == rid)
                    maj = np.bincount(pred_labels[region]).argmax()
                    pred_class_dict[int(rid)] = int(maj)

                gt_class_dict = {}
                if inst_masks.shape[0] > 0:
                    for k_, m_ in enumerate(inst_masks):
                        vals = gt_lab[m_ > 0]
                        maj_gt = np.bincount(vals).argmax() if vals.size > 0 else 0
                        gt_class_dict[int(k_ + 1)] = int(maj_gt)

                gt_cls_arr = self._sanitize_labels_to_K(
                    np.array([gt_class_dict.get(int(iid), 0) for iid in gt_ids], dtype=int), K
                )
                pr_cls_arr = self._sanitize_labels_to_K(
                    np.array([pred_class_dict.get(int(iid), 0) for iid in pr_ids], dtype=int), K
                )

                for r, c in pairs:
                    g_cls = int(gt_cls_arr[r]); p_cls = int(pr_cls_arr[c])
                    if g_cls != 0:
                        if p_cls == g_cls: per_class_TP[g_cls] += 1
                        else:
                            per_class_FN[g_cls] += 1
                            if p_cls != 0: per_class_FP[p_cls] += 1

                matched_pr_idx = set(pairs[:, 1].tolist()) if TP > 0 else set()
                for j in range(len(pr_ids)):
                    if j not in matched_pr_idx:
                        c = int(pr_cls_arr[j])
                        if c != 0: per_class_FP[c] += 1

                matched_gt_idx = set(pairs[:, 0].tolist()) if TP > 0 else set()
                for j in range(len(gt_ids)):
                    if j not in matched_gt_idx:
                        c = int(gt_cls_arr[j])
                        if c != 0: per_class_FN[c] += 1

            # --- (Optional) Save visualization ---
            if i < MAX_VIS:
                self._save_vis(
                    image=_image_from_norm(img),
                    gt_mask=gt_lab,                # show label map for consistency
                    seg_pred=pred_labels,
                    ws_mask=pred_watershed_labels,
                    name=name
                )

            # --- PQ accumulators ---
            if self.eval_mode and (compute_bPQ_and_mPQ is not None) and has_gt_masks:
                pred_class_dict_pq = {}
                for rid in np.unique(labeled_ws):
                    if rid == 0: continue
                    region = (labeled_ws == rid)
                    maj = np.bincount(pred_labels[region]).argmax()
                    pred_class_dict_pq[int(rid)] = int(maj)
                gt_class_dict_pq = {}
                if inst_masks.shape[0] > 0:
                    for k_, m_ in enumerate(inst_masks):
                        vals = gt_lab[m_ > 0]
                        maj_gt = np.bincount(vals).argmax() if vals.size > 0 else 0
                        gt_class_dict_pq[int(k_ + 1)] = int(maj_gt)

                labeled_ws_dense, pred_class_dict_dense = remap_label_and_class_map(labeled_ws, pred_class_dict_pq)
                gt_inst_dense, gt_class_dict_dense = remap_label_and_class_map(gt_inst, gt_class_dict_pq)

                gt_inst_map.append(gt_inst_dense)
                pred_inst_map.append(labeled_ws_dense)
                gt_class_map.append(gt_class_dict_dense)
                pred_class_map.append(pred_class_dict_dense)

        # ---- Aggregate (pixel metrics) ----
        out: Dict[str, float] = {}
        out["dice"] = float(np.mean(dice_scores)) if len(dice_scores) else 0.0
        conf_all = np.concatenate(ece_conf, axis=0) if len(ece_conf) else np.array([])
        corr_all = np.concatenate(ece_corr, axis=0) if len(ece_corr) else np.array([])
        out["ece"] = ece_from_conf(conf_all, corr_all, n_bins=15) if conf_all.size else 0.0
        if self.eval_mode and len(hn_dice_scores):
            out["hn_dice"] = float(np.mean(hn_dice_scores))

        # ---- Detection F1 (class-agnostic) ----
        if self.eval_mode:
            prec_d = det_TP / (det_TP + det_FP) if (det_TP + det_FP) > 0 else 0.0
            rec_d  = det_TP / (det_TP + det_FN) if (det_TP + det_FN) > 0 else 0.0
            f1_d   = 2*prec_d*rec_d / (prec_d + rec_d) if (prec_d + rec_d) > 0 else 0.0
            out["det_prec"] = float(prec_d)
            out["det_rec"]  = float(rec_d)
            out["det_f1"]   = float(f1_d)

            classes = list(range(1, K))
            per_class_f1 = {}
            f1_vals, supports = [], []
            TP_sum = FP_sum = FN_sum = 0

            for c in classes:
                TPc, FPc, FNc = per_class_TP[c], per_class_FP[c], per_class_FN[c]
                Pc  = TPc / (TPc + FPc) if (TPc + FPc) > 0 else 0.0
                Rc  = TPc / (TPc + FNc) if (TPc + FNc) > 0 else 0.0
                F1c = 2*Pc*Rc / (Pc + Rc) if (Pc + Rc) > 0 else 0.0
                cname = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                per_class_f1[cname] = float(F1c)
                f1_vals.append(F1c)
                supports.append(TPc + FNc)
                TP_sum += TPc; FP_sum += FPc; FN_sum += FNc

            macro_f1   = float(np.mean(f1_vals)) if len(f1_vals) else 0.0
            micro_prec = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
            micro_rec  = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0
            micro_f1   = 2*micro_prec*micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
            weighted_f1= float(np.average(f1_vals, weights=np.array(supports))) if sum(supports) > 0 else 0.0

            out["f1_macro"] = macro_f1
            out["f1_micro"] = float(micro_f1)
            out["f1_weighted"] = weighted_f1
            for kname, v in per_class_f1.items():
                out[f"f1_{kname}"] = v

        # ---- PQ (optional) ----
        if self.eval_mode and (compute_bPQ_and_mPQ is not None) and len(gt_inst_map) > 0:
            all_classes = list(range(K))
            bPQ, bDQ, bSQ, mPQ, pq_per_class = compute_bPQ_and_mPQ(
                gt_inst_map, pred_inst_map, gt_class_map, pred_class_map,
                all_classes, match_iou=0.5
            )
            out["bPQ"] = bPQ
            out["bDQ"] = bDQ
            out["bSQ"] = bSQ
            out["mPQ"] = mPQ
            for c in sorted(pq_per_class.keys()):
                cname = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                out[f"pq_{cname}"] = pq_per_class[c]

        print(out)
        return out

    # ---------------------------
    # Visualization
    # ---------------------------
    def _save_vis(self, image, gt_mask, seg_pred, ws_mask, name: str):
        class_colors = [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ]

        if isinstance(image, torch.Tensor):
            image = image.permute(2, 0, 1).cpu().numpy()
        image = np.clip(image, 0, 1)

        def colorize(lbl: np.ndarray) -> np.ndarray:
            lbl = lbl.astype(np.int32)
            H, W = lbl.shape
            out = np.zeros((H, W, 3), dtype=np.uint8)
            vmax = min(np.max(lbl), len(class_colors) - 1)
            for cls in range(vmax + 1):
                out[lbl == cls] = class_colors[cls]
            return out

        gt_color  = colorize(gt_mask)
        seg_color = colorize(seg_pred)
        ws_color  = colorize(ws_mask)

        fig, axs = plt.subplots(2, 2, figsize=(10, 9))
        axs[0, 0].imshow(image);    axs[0, 0].set_title("Image")
        axs[0, 1].imshow(gt_color); axs[0, 1].set_title("GT (colored)")
        axs[1, 0].imshow(seg_color);axs[1, 0].set_title("Semantic Pred")
        axs[1, 1].imshow(ws_color); axs[1, 1].set_title("Watershed Pred")
        for ax in axs.ravel(): ax.axis('off')

        plt.suptitle(f"{name} | {self.output_suffix}")
        plt.tight_layout()
        save_dir = "./final_outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f"{name}_{self.output_suffix}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[viz] {save_path}")
