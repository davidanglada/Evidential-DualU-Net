# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import torch
import torch.distributed as dist
import torchmetrics.functional as F
from torchmetrics.regression import MeanSquaredError
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os.path as osp
import os
import torchvision.transforms.v2 as v2

import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, label
from skimage import exposure
from skimage.morphology import extrema
from skimage.segmentation import watershed, find_boundaries

from typing import Dict, List, Optional, Union
from collections import OrderedDict

from ..utils.distributed import all_gather
from .pq import compute_bPQ_and_mPQ, remap_label_and_class_map
from .edl_uncert_utils import (
    predictive_entropy_from_alpha_norm,
    dirichlet_expected_entropy_norm,
    dirichlet_distributional_mi_norm,
    edl_aleatoric_scalar,
    edl_epistemic_scalar,
    edl_vacuity,
    normalize_ua_ue_tensor,
)


#########################################################
# Helper Functions
#########################################################

def pair_coordinates(
    setA: np.ndarray,
    setB: np.ndarray,
    radius: float
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Pair points between two sets using the Hungarian (Munkres) assignment algorithm,
    subject to a distance threshold ('radius'). Points in setA and setB that 
    can match within the given radius are paired.

    Returns:
        pairing (np.ndarray): (K, 2) matched indices (i in setA, j in setB).
        unpairedA (np.ndarray): Indices of setA that are unmatched.
        unpairedB (np.ndarray): Indices of setB that are unmatched.
    """
    if setA.shape[0] == 0 or setB.shape[0] == 0:
        pairing = np.empty((0, 2), dtype=np.int64)
        unpairedA = np.arange(setA.shape[0], dtype=np.int64)
        unpairedB = np.arange(setB.shape[0], dtype=np.int64)
        return pairing, unpairedA, unpairedB

    # Distance matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric="euclidean")

    # Apply Hungarian assignment
    indicesA, matchedB = linear_sum_assignment(pair_distance)
    costs = pair_distance[indicesA, matchedB]

    # Keep matches within 'radius'
    valid_matches = costs <= radius
    pairedA = indicesA[valid_matches]
    pairedB = matchedB[valid_matches]

    pairing = np.stack([pairedA, pairedB], axis=-1) if pairedA.size > 0 else np.empty((0, 2), dtype=np.int64)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)

    return pairing, unpairedA, unpairedB


def cell_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    w: List[float] = [1, 1]
) -> (float, float, float, float):
    """
    Compute detection-level F1, precision, recall, and "label matching" accuracy 
    for matched/unmatched cells.

    Returns:
        f1_d, prec_d, rec_d, acc_d
    """
    tp_d = len(paired_true)
    fp_d = len(unpaired_pred)
    fn_d = len(unpaired_true)

    # Among matched pairs, how many have the same label
    tp_tn_dt = (paired_pred == paired_true).sum()
    fp_fn_dt = (paired_pred != paired_true).sum()
    acc_d = tp_tn_dt / (tp_tn_dt + fp_fn_dt) if (tp_tn_dt + fp_fn_dt) > 0 else 0.0

    prec_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0
    rec_d  = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
    denom  = (2*tp_d + w[0]*fp_d + w[1]*fn_d)
    f1_d   = (2 * tp_d) / denom if denom > 0 else 0.0

    return f1_d, prec_d, rec_d, acc_d


def cell_type_detection_scores(
    paired_true: np.ndarray,
    paired_pred: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    type_id: int,
    w: List[float] = [2, 2, 1, 1],
    exhaustive: bool = True,
) -> (float, float, float):
    """
    Compute type-specific (class-specific) F1, precision, and recall for 
    nuclei labeled with 'type_id'.
    """
    # Only keep matched pairs where at least one is of the desired type
    type_samples = (paired_true == type_id) | (paired_pred == type_id)
    paired_true = paired_true[type_samples]
    paired_pred = paired_pred[type_samples]

    tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
    tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
    fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
    fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

    if not exhaustive:
        ignore = (paired_true == -1).sum()  # potential special-case ignoring
        fp_dt -= ignore

    fp_d = (unpaired_pred == type_id).sum()
    fn_d = (unpaired_true == type_id).sum()

    # Weighted precision / recall
    denom_prec = (tp_dt + tn_dt + w[0] * fp_dt + w[2] * fp_d)
    prec_type = (tp_dt + tn_dt) / denom_prec if denom_prec > 0 else 0.0

    denom_rec = (tp_dt + tn_dt + w[1] * fn_dt + w[3] * fn_d)
    rec_type = (tp_dt + tn_dt) / denom_rec if denom_rec > 0 else 0.0

    # Weighted F1
    denom_f1 = 2*(tp_dt + tn_dt) + w[0]*fp_dt + w[1]*fn_dt + w[2]*fp_d + w[3]*fn_d
    f1_type = (2*(tp_dt + tn_dt)) / denom_f1 if denom_f1 > 0 else 0.0

    return f1_type, prec_type, rec_type


def find_local_maxima(
    pred: np.ndarray,
    h: float,
    centers: bool = False
) -> (np.ndarray, np.ndarray):
    """
    Find local maxima in a 2D map (for centroid detection).
    If 'centers=True', assume 'pred' is already a binary map of centroids.
    Otherwise use h-maxima transform with threshold 'h'.
    """
    if not centers:
        pred_h = exposure.rescale_intensity(pred)
        h_maxima = extrema.h_maxima(pred_h, h)
    else:
        # Already a centroid map
        h_maxima = pred

    connectivity = 4
    output = cv2.connectedComponentsWithStats(h_maxima.astype(np.uint8), connectivity, ltype=cv2.CV_32S)
    num_labels = output[0]
    centroids = output[3]  # (num_labels, 2) -> [cx, cy]

    coords_list = []
    for i in range(num_labels):
        if i != 0:  # skip background
            coords_list.append((int(centroids[i, 1]), int(centroids[i, 0])))  # (y,x)

    centroid_map = np.zeros_like(h_maxima)
    kept = []
    for (r, c) in coords_list:  # Only mark if above threshold
        centroid_map[r, c] = 255
        kept.append((r, c))

    return centroid_map, np.array(kept, dtype=int)


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Traditional binary Dice coefficient. Both 'true' and 'pred' are
    converted to binary (1 => foreground, 0 => background).
    """
    t = (true > 0).astype(np.uint8)
    p = (pred > 0).astype(np.uint8)
    inter = (t * p).sum()
    denom = t.sum() + p.sum()
    return 2.0 * inter / denom if denom > 0 else 0.0


#########################################################
# Base Classes
#########################################################

class BaseCellMetric:
    """
    Base class for cell-based metrics in a (potentially) distributed environment.
    Stores predictions and targets across steps, can synchronize among processes.
    """
    def __init__(
        self,
        num_classes: int,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes
        self.thresholds = thresholds if isinstance(thresholds, list) else [thresholds]
        self.class_names = (
            class_names 
            if class_names is not None 
            else [str(i) for i in range(1, num_classes + 1)]
        )
        self.preds: List[Dict[str, torch.Tensor]] = []
        self.targets: List[Dict[str, torch.Tensor]] = []

    def synchronize_between_processes(self) -> None:
        """
        Collect predictions/targets from all processes if distributed.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.barrier()

        all_preds = all_gather(self.preds)
        all_targets = all_gather(self.targets)
        self.preds = list(itertools.chain(*all_preds))
        self.targets = list(itertools.chain(*all_targets))

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self.preds = []
        self.targets = []

    def update(
        self, 
        preds: List[Dict[str, torch.Tensor]], 
        targets: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Add new predictions and targets to internal buffers.
        Ensures centroid maps always have shape [1,H,W].
        """

        fixed_preds = []
        for p in preds:
            p_cent = p.get("centroid_prob", p.get("centroid_gaussian"))
            if p_cent.ndim == 2:
                p_cent = p_cent.unsqueeze(0)   # [H,W] → [1,H,W]

            fixed_preds.append({
                "segmentation_mask": p["segmentation_mask"],  # p_seg[i] (probs)
                "alpha_seg":         p["alpha_seg"],          # NEW: alpha_seg[i]
                "centroid_prob":     p_cent,
                "image":             p["image"],
            })

        fixed_targets = []
        for t in targets:
            t_cent = t.get("centroid_prob", t.get("centroid_gaussian"))
            if t_cent.ndim == 2:
                t_cent = t_cent.unsqueeze(0)   # [H,W] → [1,H,W]

            new_t = {
                "segmentation_mask": t["segmentation_mask"],
                "centroid_prob":     t_cent,
                "boxes":             t["boxes"],
                "labels":            t["labels"],
                "file_name":         t["file_name"],
            }
            if not self.train:
                new_t["mask"] = t["mask"]

            fixed_targets.append(new_t)

        self.preds.extend(fixed_preds)
        self.targets.extend(fixed_targets)


    def compute(self) -> Dict[str, float]:
        """
        Synchronize, gather final data, and compute the metrics.
        """
        self.synchronize_between_processes()
        values = self._get_values()
        return self._compute(*values)

    def _get_values(self):
        raise NotImplementedError

    def _compute(self, *args, **kwargs):
        raise NotImplementedError


#########################################################
# Main Multi-Task Metric
#########################################################

class MultiTaskEvaluationMetric(BaseCellMetric):
    """
    Evaluates cell segmentation, centroid detection/regression, classification, and PQ.
    
    - If train=True, we assume `t['masks']` is NOT available, skip instance-level metrics (PQ, instance-based dice, etc.).
    - If train=False, we assume `t['masks']` is available, compute additional instance-level metrics (Panoptic, etc.).
    """
    def __init__(
        self,
        num_classes: int,
        dataset: str,
        thresholds: Union[int, List[int]],
        class_names: Optional[List[str]] = None,
        max_pair_distance: float = 12,
        train: bool = True,
        th: float = 0.1,
        output_sufix: Optional[str] = None,
        # NEW: uncertainty thresholds (on [0,1] uncertainty)
        seg_unc_thr: float = 0.6,   # threshold on seg *uncertainty* (e.g. total entropy)
        cent_unc_thr: float = 1.0,  # threshold on centroid *uncertainty* ("mass")
        # NEW: Gaussian map params for centroid uncertainty
        gaussian_peak_max: float = 1.0,  # G_MAX in your snippet
        gaussian_sigma_px: float = 4.0,  # SIGMA_PX in your snippet
    ):
        super().__init__(num_classes, thresholds, class_names)
        self.max_pair_distance = max_pair_distance
        self.train = train
        self.th = th
        self.output_sufix = output_sufix if output_sufix is not None \
            else datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.dataset = dataset

        # Uncertainty-based gating thresholds:
        # keep a pair (GT, Pred) only if:
        #   seg_unc_total <= seg_unc_thr  AND  cent_unc_all <= cent_unc_thr
        self.seg_unc_thr = seg_unc_thr
        self.cent_unc_thr = cent_unc_thr
        SIGMA_PX = 5.0                        # σ of Gaussian centroid map (px)
        G_MAX    = (1.0 / (2 * math.pi * SIGMA_PX**2)) * 100.0 
        # Gaussian-map parameters used in centroid uncertainty (peak / mass / shift)
        self.gaussian_peak_max = G_MAX
        self.gaussian_sigma_px = SIGMA_PX

    def _get_values(self):
        """
        Extract necessary data from self.targets and self.preds. 
        If train=True, t['masks'] won't exist -> set them to None.
        """
        # ---- targets ----
        true_gaussian_centroids = [t["centroid_prob"].detach().cpu().numpy() for t in self.targets]
        true_labels             = [t["labels"].detach().cpu().numpy()         for t in self.targets]
        true_segmentation_mask  = [t["segmentation_mask"].detach().cpu().numpy() for t in self.targets]
        true_boxes              = [t["boxes"].detach().cpu().numpy()          for t in self.targets]
        images_names            = [t["file_name"] for t in self.targets]

        if not self.train:
            true_masks = [t["mask"].detach().cpu().numpy() for t in self.targets]
        else:
            true_masks = [None] * len(self.targets)

        # ---- predictions ----
        pred_gaussian_centroids = [p["centroid_prob"].detach().cpu().numpy()    for p in self.preds]
        pred_segmentation_mask  = [p["segmentation_mask"].detach().cpu().numpy() for p in self.preds]  # p_seg
        pred_alpha_seg          = [p["alpha_seg"].detach().cpu().numpy()        for p in self.preds]   # alphas
        images                  = [p["image"].detach().cpu().numpy()            for p in self.preds]

        # Clean memory
        for p in self.preds:
            del p["image"]
            del p["centroid_prob"]
            del p["segmentation_mask"]
            del p["alpha_seg"]          # NEW

        for t in self.targets:
            del t["centroid_prob"]
            del t["segmentation_mask"]
            del t["boxes"]
            del t["labels"]
            if not self.train:
                del t["mask"]

        torch.cuda.empty_cache()

        return (
            true_gaussian_centroids,
            true_labels,
            true_segmentation_mask,
            true_boxes,
            pred_gaussian_centroids,
            pred_segmentation_mask,
            pred_alpha_seg,       # NEW in the tuple
            images,
            true_masks,
            images_names,
        )


    def _compute(
        self,
        true_gaussian_centroids: List[np.ndarray],
        true_labels: List[np.ndarray],
        true_segmentation_mask: List[np.ndarray],
        true_boxes: List[np.ndarray],
        pred_gaussian_centroids: List[np.ndarray],
        pred_segmentation_mask: List[np.ndarray],  # p_seg
        pred_alpha_seg: List[np.ndarray],          # NEW
        images: List[np.ndarray],
        true_masks: List[Optional[np.ndarray]],
        images_names: List[str]
    ) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}

        MAX_VIS = 20

        dice_scores = []
        mse_scores = []

        conf_all = []
        corr_all = []

        # --- RAW detection accumulators (no uncertainty filtering) ---
        paired_all_raw = []
        unpaired_true_all_raw = []
        unpaired_pred_all_raw = []

        # --- FILTERED detection accumulators (confidence-gated pairs) ---
        paired_all_filt = []
        unpaired_true_all_filt = []  # same as raw; we only filter pairs
        unpaired_pred_all_filt = []

        # shared label pools
        true_inst_type_all = []
        pred_inst_type_all = []
        true_idx_offset = 0
        pred_idx_offset = 0

        if not self.train:
            hn_dice_scores = []
            gt_inst_map = []
            pred_inst_map = []
            gt_class_map = []
            pred_class_map = []
        else:
            hn_dice_scores = []
            gt_inst_map = []
            pred_inst_map = []
            gt_class_map = []
            pred_class_map = []

        # ---------------- Main Loop ---------------- #
        for i in range(len(true_gaussian_centroids)):
            t_gauss = true_gaussian_centroids[i]
            p_gauss = pred_gaussian_centroids[i]
            t_seg   = true_segmentation_mask[i]
            t_seg_colors = t_seg.copy()
            p_seg   = pred_segmentation_mask[i]   # [K,H,W] probs
            a_seg   = pred_alpha_seg[i]           # [K,H,W] alphas
            image_name = images_names[i]

            # GT centroids from boxes
            boxes_i = true_boxes[i]
            labels_i = true_labels[i]
            true_cents_i = []
            for box in boxes_i:
                x0, y0, w, h = box
                cx = x0 + w // 2
                cy = y0 + h // 2
                true_cents_i.append((cy, cx))
            true_cents_i = np.array(true_cents_i)

            # Dice & MSE
            d_i = self._dice_coefficient(
                t_seg,
                pred_masks=np.argmax(p_seg, axis=0),
                num_classes=self.num_classes
            )
            dice_scores.append(d_i)

            m_i = self._mse_centroids(t_gauss, p_gauss)
            mse_scores.append(m_i)

            # ECE data
            flat_prob = p_seg.max(axis=0).ravel()
            flat_pred = p_seg.argmax(axis=0).ravel()
            flat_true = t_seg.ravel()
            conf_all.append(flat_prob)
            corr_all.append((flat_pred == flat_true).astype(np.float32))

            # Watershed + per-instance confidences
            (
                pred_cents_i,
                pred_labels_i,
                watershed_mask,
                cells_mask,
                seg_unc_i_dict,
                cent_unc_i_dict,
            ) = self._perform_watershed(
                p_seg=p_seg,
                alpha_seg=a_seg,
                pred_centroids=p_gauss,
            )

            # HN dice
            pred_binary = (watershed_mask > 0).astype(np.uint8)
            true_binary = (t_seg > 0).astype(np.uint8)
            cc_pred, _ = label(pred_binary)
            cc_true, _ = label(true_binary)
            hn_dice = get_dice_1(cc_true, cc_pred)
            hn_dice_scores.append(hn_dice)

            # Empty-handling as in your original code
            if true_cents_i.shape[0] == 0:
                true_cents_i = np.array([[0, 0]])
                labels_i = np.array([0])
            if pred_cents_i.shape[0] == 0:
                pred_cents_i = np.array([[0, 0]])
                pred_labels_i = np.array([0])

            # Pairing (GT vs Pred)
            paired, unpaired_true, unpaired_pred = pair_coordinates(
                true_cents_i, pred_cents_i, self.max_pair_distance
            )

                        # --- UNCERTAINTY GATE over the pairs (for filtered version) ---
            seg_unc_total_i = seg_unc_i_dict["epi"]  # [N] per instance
            cent_unc_all_i  = cent_unc_i_dict["mass"]     # [N] per instance

            if paired.shape[0] > 0 and seg_unc_total_i.size > 0 and cent_unc_all_i.size > 0:
                pred_idx_for_pairs = paired[:, 1]

                # safety: ensure indices do not go out of bounds
                valid = (
                    (pred_idx_for_pairs < seg_unc_total_i.shape[0]) &
                    (pred_idx_for_pairs < cent_unc_all_i.shape[0])
                )
                if np.any(valid):
                    pred_idx_valid = pred_idx_for_pairs[valid]
                    paired_valid   = paired[valid]

                    seg_unc_pairs  = seg_unc_total_i[pred_idx_valid]
                    cent_unc_pairs = cent_unc_all_i[pred_idx_valid]

                    conf_mask = (
                        (seg_unc_pairs <= self.seg_unc_thr) &
                        (cent_unc_pairs <= self.cent_unc_thr)
                    )
                    paired_kept = paired_valid[conf_mask]
                else:
                    # all indices invalid → drop all for filtered metrics
                    paired_kept = np.empty((0, 2), dtype=paired.dtype)
            else:
                # no pairs or no uncertainty info → nothing kept for filtered
                paired_kept = np.empty((0, 2), dtype=paired.dtype)


            # --------- Accumulate with offsets --------- #
            true_idx_offset = (
                true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
            )
            pred_idx_offset = (
                pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
            )

            true_inst_type_all.append(labels_i)
            pred_inst_type_all.append(pred_labels_i)

            # local copies
            unpaired_true_i = unpaired_true.copy()
            unpaired_pred_i = unpaired_pred.copy()
            paired_raw_i = paired.copy()
            paired_filt_i = paired_kept.copy()

            # RAW pairs
            if paired_raw_i.shape[0] != 0:
                paired_raw_i[:, 0] += true_idx_offset
                paired_raw_i[:, 1] += pred_idx_offset
                paired_all_raw.append(paired_raw_i)

            # FILTERED pairs (only confident pairs)
            if paired_filt_i.shape[0] != 0:
                paired_filt_i[:, 0] += true_idx_offset
                paired_filt_i[:, 1] += pred_idx_offset
                paired_all_filt.append(paired_filt_i)

            # Unpaired are the same for raw & filtered (we only ignore some pairs)
            unpaired_true_i += true_idx_offset
            unpaired_pred_i += pred_idx_offset
            unpaired_true_all_raw.append(unpaired_true_i)
            unpaired_pred_all_raw.append(unpaired_pred_i)
            unpaired_true_all_filt.append(unpaired_true_i)
            unpaired_pred_all_filt.append(unpaired_pred_i)

            # Instance-level PQ, etc. (unchanged)
            if not self.train and true_masks[i] is not None:
                true_inst_masks_i = true_masks[i]

                cc_pred = label((watershed_mask > 0).astype(np.uint8))[0]
                cc_true = np.zeros_like(cc_pred, dtype=np.int32)
                for k in range(true_inst_masks_i.shape[0]):
                    cc_true[true_inst_masks_i[k] > 0] = k + 1

                gt_class_map_i = {k + 1: lab for k, lab in enumerate(labels_i)}
                pred_class_map_i = {k + 1: lab for k, lab in enumerate(pred_labels_i)}

                cc_pred, pred_class_map_i = remap_label_and_class_map(cc_pred, pred_class_map_i)
                cc_true, gt_class_map_i   = remap_label_and_class_map(cc_true, gt_class_map_i)

                gt_inst_map.append(cc_true)
                pred_inst_map.append(cc_pred)
                gt_class_map.append(gt_class_map_i)
                pred_class_map.append(pred_class_map_i)

                if i < MAX_VIS:
                    self._save_visualization(
                        image=self._get_raw_image(images[i]),
                        gt_mask=t_seg_colors,
                        segmentation_mask=p_seg.argmax(axis=0),
                        true_centroids_list=true_cents_i,
                        pred_centroids_list=pred_cents_i,
                        w_centroids_list=pred_cents_i,
                        classification_mask=p_seg,
                        watershed_mask=watershed_mask,
                        true_gaussian=t_gauss[0],
                        pred_gaussian=p_gauss[0],
                        cells_mask=cells_mask,
                        true_labels=labels_i,
                        pred_labels=pred_labels_i,
                        mse=m_i,
                        hn_dice=hn_dice,
                        detection_f1=0.0,
                        class_f1_scores={},
                        filename_prefix=image_name,
                        output_sufix=self.output_sufix,
                        dataset=self.dataset
                    )

        # ---------- Flatten detection pools ---------- #
        def _safe_concat(list_of_arrays, shape):
            if len(list_of_arrays) == 0:
                return np.empty(shape, dtype=np.int64)
            return np.concatenate(list_of_arrays, axis=0)

        paired_all_raw  = _safe_concat(paired_all_raw,  (0, 2))
        paired_all_filt = _safe_concat(paired_all_filt, (0, 2))

        unpaired_true_all_raw  = _safe_concat(unpaired_true_all_raw,  (0,))
        unpaired_pred_all_raw  = _safe_concat(unpaired_pred_all_raw,  (0,))
        unpaired_true_all_filt = _safe_concat(unpaired_true_all_filt, (0,))
        unpaired_pred_all_filt = _safe_concat(unpaired_pred_all_filt, (0,))

        true_inst_type_all = _safe_concat(true_inst_type_all, (0,))
        pred_inst_type_all = _safe_concat(pred_inst_type_all, (0,))

        # --- RAW view ---
        if paired_all_raw.shape[0] > 0:
            paired_true_type_raw = true_inst_type_all[paired_all_raw[:, 0]]
            paired_pred_type_raw = pred_inst_type_all[paired_all_raw[:, 1]]
        else:
            paired_true_type_raw = np.empty((0,), dtype=np.int64)
            paired_pred_type_raw = np.empty((0,), dtype=np.int64)

        unpaired_true_type_raw = true_inst_type_all[unpaired_true_all_raw] if unpaired_true_all_raw.size > 0 else np.empty((0,), dtype=np.int64)
        unpaired_pred_type_raw = pred_inst_type_all[unpaired_pred_all_raw] if unpaired_pred_all_raw.size > 0 else np.empty((0,), dtype=np.int64)

        # --- FILTERED view ---
        if paired_all_filt.shape[0] > 0:
            paired_true_type_filt = true_inst_type_all[paired_all_filt[:, 0]]
            paired_pred_type_filt = pred_inst_type_all[paired_all_filt[:, 1]]
        else:
            paired_true_type_filt = np.empty((0,), dtype=np.int64)
            paired_pred_type_filt = np.empty((0,), dtype=np.int64)

        unpaired_true_type_filt = true_inst_type_all[unpaired_true_all_filt] if unpaired_true_all_filt.size > 0 else np.empty((0,), dtype=np.int64)
        unpaired_pred_type_filt = pred_inst_type_all[unpaired_pred_all_filt] if unpaired_pred_all_filt.size > 0 else np.empty((0,), dtype=np.int64)

        # ---------- Overall detection metrics ---------- #
        # RAW
        f1_d_raw, prec_d_raw, rec_d_raw, acc_d_raw = cell_detection_scores(
            paired_true=paired_true_type_raw,
            paired_pred=paired_pred_type_raw,
            unpaired_true=unpaired_true_type_raw,
            unpaired_pred=unpaired_pred_type_raw
        )

        # FILTERED (uncertainty-aware)
        f1_d_filt, prec_d_filt, rec_d_filt, acc_d_filt = cell_detection_scores(
            paired_true=paired_true_type_filt,
            paired_pred=paired_pred_type_filt,
            unpaired_true=unpaired_true_type_filt,
            unpaired_pred=unpaired_pred_type_filt
        )

        nuclei_metrics = {
            # keep backwards-compatible key as RAW
            "detection": {
                "f1":   f1_d_raw,
                "prec": prec_d_raw,
                "rec":  rec_d_raw,
                "acc":  acc_d_raw,
            },
            # new filtered variant
            "detection_filt": {
                "f1":   f1_d_filt,
                "prec": prec_d_filt,
                "rec":  rec_d_filt,
                "acc":  acc_d_filt,
            },
        }

        # ---------- Class-wise detection ---------- #
        if self.num_classes > 1:
            for nuc_type in range(1, self.num_classes + 1):
                cname = self.class_names[nuc_type - 1]

                # RAW
                f1_cell_raw, prec_cell_raw, rec_cell_raw = cell_type_detection_scores(
                    paired_true_type_raw,
                    paired_pred_type_raw,
                    unpaired_true_type_raw,
                    unpaired_pred_type_raw,
                    nuc_type
                )
                nuclei_metrics[cname] = {
                    "f1":   f1_cell_raw,
                    "prec": prec_cell_raw,
                    "rec":  rec_cell_raw,
                }

                # FILTERED
                f1_cell_filt, prec_cell_filt, rec_cell_filt = cell_type_detection_scores(
                    paired_true_type_filt,
                    paired_pred_type_filt,
                    unpaired_true_type_filt,
                    unpaired_pred_type_filt,
                    nuc_type
                )
                nuclei_metrics[f"{cname}_filt"] = {
                    "f1":   f1_cell_filt,
                    "prec": prec_cell_filt,
                    "rec":  rec_cell_filt,
                }

        # ---------- Aggregate basic metrics ---------- #
        all_metrics = nuclei_metrics
        all_metrics["dice"] = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
        all_metrics["mse"]  = float(np.mean(mse_scores))  if len(mse_scores) > 0 else 0.0

        conf_flat = np.concatenate(conf_all, axis=0)
        corr_flat = np.concatenate(corr_all, axis=0)
        ece_value = self._ece_from_conf(conf_flat, corr_flat, n_bins=15)
        all_metrics["ece"] = ece_value

        # ---------- PQ etc. (unchanged) ---------- #
        if not self.train and len(gt_inst_map) > 0:
            hn_dice_mean = float(np.mean(hn_dice_scores)) if len(hn_dice_scores) > 0 else 0.0
            all_metrics["hn_dice"] = hn_dice_mean

            all_classes = list(range(1, self.num_classes + 1))
            bPQ, bDQ, bSQ, mPQ, pq_per_class = compute_bPQ_and_mPQ(
                gt_inst_map,
                pred_inst_map,
                gt_class_map,
                pred_class_map,
                all_classes,
                match_iou=0.5
            )
            all_metrics["bPQ"] = bPQ
            all_metrics["bDQ"] = bDQ
            all_metrics["bSQ"] = bSQ
            all_metrics["mPQ"] = mPQ

            for c in sorted(pq_per_class.keys()):
                class_idx = c - 1
                class_key = (
                    self.class_names[class_idx]
                    if class_idx < len(self.class_names)
                    else f"class_{c}"
                )
                all_metrics[f"pq_{class_key}"] = pq_per_class[c]

        print(all_metrics)
        return all_metrics


    ##################################################
    # Internal Utility Methods
    ##################################################

    def _dice_coefficient(
        self,
        true_masks: Union[torch.Tensor, np.ndarray],
        pred_masks: Union[torch.Tensor, np.ndarray],
        num_classes: int
    ) -> float:
        """
        Mean Dice across 'num_classes'.
        """
        if not isinstance(true_masks, torch.Tensor):
            true_masks = torch.tensor(true_masks)
        if not isinstance(pred_masks, torch.Tensor):
            pred_masks = torch.tensor(pred_masks)

        mean_dice = F.dice(pred_masks, true_masks.int())
        return float(mean_dice.item())

    def _mse_centroids(
        self,
        true_gaussian_mask: Union[torch.Tensor, np.ndarray],
        pred_gaussian_mask: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """
        Compute MSE between ground-truth and predicted centroid heatmaps (1,H,W).
        """
        if not isinstance(true_gaussian_mask, torch.Tensor):
            true_gaussian_mask = torch.tensor(true_gaussian_mask)
        if not isinstance(pred_gaussian_mask, torch.Tensor):
            pred_gaussian_mask = torch.tensor(pred_gaussian_mask)

        mse_metric = MeanSquaredError()
        val = mse_metric(pred_gaussian_mask, true_gaussian_mask)
        return float(val.item())
    
    def _ece_from_conf(
        self,
        conf: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 15) -> float:
        """
        Args
        ----
        conf    : 1-D array of predicted confidences ∈ [0,1]
        correct : 1-D boolean array –  True if prediction was correct
        Returns
        -------
        scalar ECE  (0 = perfect)
        """
        edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi)
            if m.any():
                acc_bin  = correct[m].mean()
                conf_bin = conf[m].mean()
                ece     += m.mean() * abs(acc_bin - conf_bin)
        return float(ece)

    def _perform_watershed(
        self,
        p_seg: np.ndarray,        # [K,H,W] mean probabilities (for argmax)
        alpha_seg: np.ndarray,    # [K,H,W] Dirichlet alphas (for EDL)
        pred_centroids: np.ndarray  # [1,H,W] Gaussian-like map
        ) -> (
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Dict[str, np.ndarray],
            Dict[str, np.ndarray],
        ):
        """
        Perform watershed with predicted centroid markers.

        Returns:
          predicted_centroids (N,2) [y,x],
          predicted_classes   (N,),
          predicted_mask      (H,W) as majority-class map,
          cells_mask          (H,W) binary foreground,
          seg_uncert          dict of 1D arrays per instance:
                               {
                                 'total':   [N],  # predictive entropy / log K
                                 'ale':     [N],  # normalized UA
                                 'epi':     [N],  # normalized UE
                                 'vacuity': [N],  # mean vacuity
                               }
          cen_uncert          dict of 1D arrays per instance:
                               {
                                 'peak':  [N],
                                 'mass':  [N],
                                 'shift': [N],
                                 'all':   [N],
                               }
        """
        # --- Step 1: centroid markers from Gaussian map ---
        centroid_mask, _ = find_local_maxima(pred_centroids[0], self.th)
        _, markers = cv2.connectedComponents(centroid_mask.astype(np.uint8), 4, ltype=cv2.CV_32S)
        
        # Argmax segmentation for watershed & class labels
        pred_mask_argmax = np.argmax(p_seg, axis=0).astype(np.uint8)
        cells_mask = np.zeros_like(pred_mask_argmax, dtype=np.uint8)
        cells_mask[pred_mask_argmax > 0] = 1

        # --- Step 2: Watershed segmentation ---
        distance_map = distance_transform_edt(cells_mask)
        watershed_result = watershed(-distance_map, markers, mask=cells_mask, compactness=1)

        # Remove outer boundaries
        contours = np.invert(find_boundaries(watershed_result, mode='outer', background=0))
        watershed_result = watershed_result * contours

        binary_mask = np.zeros_like(watershed_result, dtype=np.uint8)
        binary_mask[watershed_result > 0] = 1
        predicted_mask = pred_mask_argmax * binary_mask

        # Relabel watershed regions
        labeled_mask, num_labels = label(watershed_result)

        # --- Step 3: precompute seg EDL uncertainties per pixel ---
        alpha = torch.from_numpy(alpha_seg).float()     # [K,H,W]
        K = alpha.shape[0]
        alpha_lastK = alpha.permute(1, 2, 0)            # [H,W,K]

        ua = edl_aleatoric_scalar(alpha_lastK)          # [H,W]
        ue = edl_epistemic_scalar(alpha_lastK)          # [H,W]
        vac = edl_vacuity(alpha_lastK)                  # [H,W]
        ua_n, ue_n = normalize_ua_ue_tensor(ua, ue, K)

        Htot = predictive_entropy_from_alpha_norm(alpha_lastK)  # [H,W]

        ua_n_np  = ua_n.cpu().numpy()
        ue_n_np  = ue_n.cpu().numpy()
        vac_np   = vac.cpu().numpy()
        Htot_np  = Htot.cpu().numpy()

        # --- Step 4: per-instance uncertainties ---
        predicted_centroids = []
        predicted_classes   = []

        seg_uncert = {
            "total":   [],
            "ale":     [],
            "epi":     [],
            "vacuity": [],
        }
        cen_uncert = {
            "peak":  [],
            "mass":  [],
            "shift": [],
            "all":   [],
        }

        g_full = pred_centroids[0]  # [H,W]

        for rid in np.unique(labeled_mask):
            if rid == 0:
                continue
            region = (labeled_mask == rid)
            if not region.any():
                continue

            # Majority class in region from probabilities
            class_in_region = pred_mask_argmax[region]
            if class_in_region.size == 0:
                continue
            majority_class = np.bincount(class_in_region).argmax()
            predicted_mask[region] = majority_class

            # centroid (y,x)
            coords_region = np.argwhere(region)
            centroid_yx = coords_region.mean(axis=0)
            cy, cx = centroid_yx[0], centroid_yx[1]
            predicted_centroids.append((int(cy), int(cx)))
            predicted_classes.append(majority_class)

            # ---------- SEG instance uncertainties ----------
            ua_reg   = ua_n_np[region]
            ue_reg   = ue_n_np[region]
            vac_reg  = vac_np[region]
            Htot_reg = Htot_np[region]

            seg_uncert["ale"].append(float(ua_reg.mean())   if ua_reg.size   > 0 else 1.0)
            seg_uncert["epi"].append(float(ue_reg.mean())   if ue_reg.size   > 0 else 1.0)
            seg_uncert["vacuity"].append(float(vac_reg.mean()) if vac_reg.size > 0 else 1.0)
            seg_uncert["total"].append(float(Htot_reg.mean())  if Htot_reg.size > 0 else 1.0)

            # ---------- CENTROID instance uncertainties ----------
            region_vals = g_full[region]
            if region_vals.size > 0:
                peak_val = float(region_vals.max())
                S_i      = float(region_vals.sum())
            else:
                peak_val = 0.0
                S_i      = 0.0

            peak_norm = peak_val / max(self.gaussian_peak_max, 1e-8)
            peak_norm = np.clip(peak_norm, 0.0, 1.0)
            u_peak = 1.0 - peak_norm

            u_mass = abs(1.0 - S_i / 100.0)
            u_mass = np.clip(u_mass, 0.0, 1.0)

            cm_rc = coords_region.mean(axis=0)
            peak_coords = coords_region[region_vals.argmax()] if region_vals.size > 0 else cm_rc
            d_px = float(np.linalg.norm(peak_coords - cm_rc))
            u_shift = np.tanh(d_px / self.gaussian_sigma_px)

            u_cent_inst = 0.3 * u_peak + 0.6 * u_mass + 0.0 * u_shift
            u_cent_inst = float(np.clip(u_cent_inst, 0.0, 1.0))

            cen_uncert["peak"].append(u_peak)
            cen_uncert["mass"].append(u_mass)
            cen_uncert["shift"].append(u_shift)
            cen_uncert["all"].append(u_cent_inst)

        # Convert lists to arrays
        predicted_centroids = np.asarray(predicted_centroids, dtype=int) if len(predicted_centroids) > 0 else np.zeros((0, 2), dtype=int)
        predicted_classes   = np.asarray(predicted_classes,   dtype=int) if len(predicted_classes)   > 0 else np.zeros((0,), dtype=int)

        for k in seg_uncert:
            seg_uncert[k] = np.asarray(seg_uncert[k], dtype=np.float32)
        for k in cen_uncert:
            cen_uncert[k] = np.asarray(cen_uncert[k], dtype=np.float32)

        return predicted_centroids, predicted_classes, predicted_mask, cells_mask, seg_uncert, cen_uncert



    def _denormalize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float]  = [0.229, 0.224, 0.225]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize image using (mean, std)."""
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

    def _get_raw_image(self, img: np.ndarray) -> torch.Tensor:
        """Convert raw image array (C,H,W) from normalization to a float32 tensor in [0,1]."""
        img = self._denormalize(img)
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        return transforms(img)

    def _save_visualization(
        self,
        image: Union[np.ndarray, torch.Tensor],
        gt_mask: np.ndarray,
        segmentation_mask: np.ndarray,
        true_centroids_list: np.ndarray,
        pred_centroids_list: np.ndarray,
        w_centroids_list: np.ndarray,
        classification_mask: np.ndarray,
        watershed_mask: np.ndarray,
        true_gaussian: np.ndarray,
        pred_gaussian: np.ndarray,
        cells_mask: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        mse: float,
        hn_dice: float,
        detection_f1: float,
        class_f1_scores: Dict[str, float],
        filename_prefix: str = "output",
        output_sufix: str = "output",
        dataset: str = "pannuke"
    ) -> None:
        """
        Save debug visualizations including original image, ground truth mask, watershed mask,
        Gaussian maps, and various metric info. Color mappings and legends are set based on the dataset.
        """
        import matplotlib.pyplot as plt

        # Set color maps and legend elements based on dataset
        if dataset == "consep":
            class_colors = [
                [0, 0, 0],       # Background: Black
                [255, 0, 0],     # Miscellaneous: Red
                [0, 255, 0],     # Inflammatory: Green
                [0, 0, 255],     # Epithelial: Blue
                [255, 255, 0]    # Spindleshaped: Yellow
            ]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Miscellaneous', markerfacecolor='r', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Inflammatory', markerfacecolor='g', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Epithelial', markerfacecolor='b', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Spindleshaped', markerfacecolor='y', markersize=10)
            ]
        elif dataset == "ki67":
            class_colors = [
                [0, 0, 0],       # Background: Black
                [255, 0, 0],     # Class 1: Red
                [0, 255, 0],     # Class 2: Green
                [0, 0, 255]      # Class 3: Blue
            ]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Class1', markerfacecolor='r', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Class2', markerfacecolor='g', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Class3', markerfacecolor='b', markersize=10)
            ]
        elif dataset == "pannuke":
            class_colors = [
                [0, 0, 0],       # Background: Black
                [255, 0, 0],     # Neoplastic: Red
                [0, 255, 0],     # Inflammatory: Green
                [255, 255, 0],   # Connective: Yellow
                [255, 255, 255], # Necrosis: White
                [0, 0, 255]      # Epithelial: Blue
            ]
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='Neoplastic', markerfacecolor='r', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Inflammatory', markerfacecolor='g', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Connective', markerfacecolor='y', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Necrosis', markerfacecolor='w', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Epithelial', markerfacecolor='b', markersize=10)
            ]
        else:
            class_colors = None
            legend_elements = None

        # Convert image to numpy array if it's a torch.Tensor.
        if isinstance(image, torch.Tensor):
            image = image.permute(2, 0, 1).cpu().numpy()
        image = np.clip(image, 0, 1)

        # convert to int
        gt_labels = gt_mask.astype(np.int32)
        pred_labels = watershed_mask
        
        if class_colors is not None:
            gt_color = np.zeros((gt_labels.shape[0], gt_labels.shape[1], 3), dtype=np.uint8)
            pred_color = np.zeros((pred_labels.shape[0], pred_labels.shape[1], 3), dtype=np.uint8)
            seg_color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
            for cls in range(len(class_colors)):
                print(cls)
                gt_color[gt_labels == cls] = class_colors[cls]
                pred_color[pred_labels == cls] = class_colors[cls]
                seg_color[segmentation_mask == cls] = class_colors[cls]
        else:
            gt_color = gt_labels
            pred_color = pred_labels
            seg_color = segmentation_mask

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Original Image")

        axs[0, 1].imshow(gt_color)
        axs[0, 1].set_title("GT Mask (Colored)")
        if legend_elements is not None:
            axs[0, 1].legend(handles=legend_elements, loc="upper right")

        axs[0, 2].imshow(seg_color)
        axs[0, 2].set_title("Predicted Segmentation")

        # Show watershed mask as predicted segmentation
        axs[1, 2].imshow(pred_color)
        axs[1, 2].set_title("Watershed Mask (Colored)")

        axs[1, 0].imshow(true_gaussian, cmap="jet")
        axs[1, 0].set_title("True Gaussian")

        axs[1, 1].imshow(pred_gaussian, cmap="jet")
        axs[1, 1].set_title("Pred Gaussian")

        plt.suptitle(f"{filename_prefix} | {output_sufix}")
        plt.tight_layout()
        save_dir = "./final_outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f"{filename_prefix}_{output_sufix}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Visualization saved to {save_path}")
