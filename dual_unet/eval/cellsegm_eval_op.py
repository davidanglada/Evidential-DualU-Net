# -*- coding: utf-8 -*-
"""metrics_updated.py – Memory-efficient metrics with exact centroid logic & visualization
=========================================================================================
Provides:
    • BaseCellMetric            – distributed-aware streaming base class
    • MultiTaskEvaluationMetric – streaming Dice, MSE, ECE, detection F1,
      per-class F1, optional visualization & PQ in test mode
"""

from __future__ import annotations
import datetime
import os
import os.path as osp
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import cv2
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt, label as nd_label
from skimage import exposure
from skimage.morphology import extrema
from skimage.segmentation import watershed, find_boundaries

from ..utils.distributed import is_dist_avail_and_initialized, all_gather
from .pq import compute_bPQ_and_mPQ, remap_label_and_class_map


################################################################################
# ----------------------- helper functions (unchanged) ------------------------#
################################################################################

def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    t = (true > 0).astype(np.uint8)
    p = (pred > 0).astype(np.uint8)
    inter = (t & p).sum()
    denom = t.sum() + p.sum()
    return float(2.0 * inter / (denom + 1e-6))


def find_local_maxima(pred: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    pr = exposure.rescale_intensity(pred)
    hmax = extrema.h_maxima(pr, h)
    num, _, _, cents = cv2.connectedComponentsWithStats(hmax.astype(np.uint8), 4, cv2.CV_32S)
    cmap = np.zeros_like(hmax, dtype=np.uint8)
    coords: List[Tuple[int,int]] = []
    for i in range(1, num):
        r, c = int(cents[i,1]), int(cents[i,0])
        coords.append((r, c))
        cmap[r, c] = 255
    return cmap, np.asarray(coords, dtype=np.int64)


def pair_coordinates(
    A: np.ndarray, B: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist_mat = scipy.spatial.distance.cdist(A, B, metric="euclidean")
    iA, iB = linear_sum_assignment(dist_mat)
    costs = dist_mat[iA, iB]
    keep = costs <= radius
    paired = np.column_stack([iA[keep], iB[keep]])
    unA = np.delete(np.arange(len(A)), paired[:,0]) if paired.size else np.arange(len(A))
    unB = np.delete(np.arange(len(B)), paired[:,1]) if paired.size else np.arange(len(B))
    return paired, unA, unB


def cell_detection_scores(
    p_t: np.ndarray, p_p: np.ndarray, u_t: np.ndarray, u_p: np.ndarray,
    w: List[float]=[1.0,1.0]
) -> Tuple[float,float,float,float]:
    tp = len(p_p)
    fp = len(u_p)
    fn = len(u_t)
    corr = (p_t==p_p).sum()
    inc  = (p_t!=p_p).sum()
    acc  = corr/(corr+inc+1e-6)
    prec = tp/(tp+fp+1e-6)
    rec  = tp/(tp+fn+1e-6)
    f1   = 2*tp/(2*tp + w[0]*fp + w[1]*fn + 1e-6)
    return f1, prec, rec, acc


def cell_type_detection_scores(
    p_t: np.ndarray, p_p: np.ndarray, u_t: np.ndarray, u_p: np.ndarray,
    type_id: int, w: List[int]=[2,2,1,1], exhaustive: bool=True
) -> Tuple[float,float,float]:
    mask = (p_t==type_id)|(p_p==type_id)
    pt = p_t[mask]; pp = p_p[mask]
    tp_dt = ((pt==type_id)&(pp==type_id)).sum()
    tn_dt = ((pt!=type_id)&(pp!=type_id)).sum()
    fp_dt = ((pt!=type_id)&(pp==type_id)).sum()
    fn_dt = ((pt==type_id)&(pp!=type_id)).sum()
    if not exhaustive:
        fp_dt -= (pt==-1).sum()
    fp_d = (u_p==type_id).sum()
    fn_d = (u_t==type_id).sum()
    prec = (tp_dt+tn_dt)/(tp_dt+tn_dt + w[0]*fp_dt + w[2]*fp_d + 1e-6)
    rec  = (tp_dt+tn_dt)/(tp_dt+tn_dt + w[1]*fn_dt + w[3]*fn_d + 1e-6)
    f1   = 2*(tp_dt+tn_dt)/(2*(tp_dt+tn_dt)+w[0]*fp_dt+w[1]*fn_dt+w[2]*fp_d+w[3]*fn_d+1e-6)
    return float(f1), float(prec), float(rec)


################################################################################
# ------------------------ BaseCellMetric class ------------------------------#
################################################################################

class BaseCellMetric:
    """Distributed-aware streaming metric base class."""
    def __init__(self, num_classes:int, class_names:Optional[List[str]]=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(1,num_classes+1)]
        self.reset()

    def reset(self):
        self.n = 0
        self.dice_sum = 0.0
        self.mse_sum  = 0.0
        self.hn_sum   = 0.0
        self.tp = self.fp = self.fn = 0
        self.tp_tn = self.fp_fn = 0
        self.tp_cls = defaultdict(int)
        self.fp_cls = defaultdict(int)
        self.fn_cls = defaultdict(int)
        self.nbins     = 15
        self.bin_edges = np.linspace(0.0, 1.0, self.nbins+1, dtype=np.float32)
        self.bin_conf  = np.zeros(self.nbins, dtype=np.float64)
        self.bin_corr  = np.zeros(self.nbins, dtype=np.float64)
        self.bin_count = np.zeros(self.nbins, dtype=np.int64)
        self.pixels    = 0
        self.preds:   List[Dict[str,Any]] = []
        self.targets: List[Dict[str,Any]] = []

    def _to_cpu_np(self, x:torch.Tensor, fp16:bool=False)->np.ndarray:
        if x.is_floating_point() and fp16:
            x = x.half()
        if not x.is_floating_point():
            x = x.to(torch.int16 if x.max()<2**15 else torch.int32)
        return x.detach().cpu().numpy()

    def _all_reduce(self, t:torch.Tensor)->torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def synchronize_between_processes(self):
        if is_dist_avail_and_initialized():
            dist.barrier()
            self.preds   = list(itertools.chain(*all_gather(self.preds)))
            self.targets = list(itertools.chain(*all_gather(self.targets)))

    def update(
        self,
        preds: Sequence[Dict[str,torch.Tensor]],
        targets: Sequence[Dict[str,torch.Tensor]]
    ) -> None:
        for p,t in zip(preds, targets):
            # Dice
            pm = p["segmentation_mask"].argmax(0); tm = t["segmentation_mask"]
            self.dice_sum += get_dice_1(tm.cpu().numpy(), pm.cpu().numpy())
            # MSE
            self.mse_sum  += F.mse_loss(
                p["centroid_gaussian"].float(),
                t["centroid_gaussian"].unsqueeze(0).float(),
                reduction="mean"
            ).item()
            # Watershed + HN-dice
            pm_np = self._to_cpu_np(p["segmentation_mask"])
            cg_np = self._to_cpu_np(p["centroid_gaussian"])
            tm_np = tm.cpu().numpy()
            cmap,_ = find_local_maxima(cg_np[0], h=self.th)
            _, markers = cv2.connectedComponents(cmap, 4, cv2.CV_32S)
            cells = (pm_np.argmax(0)>0).astype(np.uint8)
            distm = distance_transform_edt(cells)
            ws = watershed(-distm, markers, mask=cells, compactness=1)
            ws *= np.invert(find_boundaries(ws,mode="outer",background=0))
            self.hn_sum += get_dice_1(tm_np>0, ws>0)
            # True centroids
            boxes = t["boxes"].cpu().numpy(); labels = t["labels"].cpu().numpy()
            true_list = [(y0+h//2, x0+w//2) for (x0,y0,w,h) in boxes]
            true_c = np.asarray(true_list or [(0,0)],dtype=np.float32)
            labels = labels if labels.size>0 else np.array([0],dtype=np.int64)
            # Predicted centroids & classes
            lab,_ = nd_label(ws)
            pred_list, pred_cls = [], []
            for rid in np.unique(lab):
                if rid==0: continue
                maskr = lab==rid
                coords = np.argwhere(maskr); cy,cx = coords.mean(0)
                maj = np.bincount(pm_np.argmax(0)[maskr]).argmax()
                pred_list.append((cy,cx)); pred_cls.append(int(maj))
            pred_c = np.asarray(pred_list or [(0,0)],dtype=np.float32)
            pred_cl= np.asarray(pred_cls or [0],dtype=np.int64)
            # Pairing
            paired,unT,unP = pair_coordinates(true_c,pred_c,self.max_pair_distance)
            # Global detection
            self.tp+=paired.shape[0]; self.fp+=len(unP); self.fn+=len(unT)
            if paired.size:
                pt = labels[paired[:,0]]; pp = pred_cl[paired[:,1]]
                self.tp_tn+=(pt==pp).sum(); self.fp_fn+=(pt!=pp).sum()
            else:
                pt=np.array([],dtype=np.int64); pp=np.array([],dtype=np.int64)
            # Per-class
            for cid in range(1,self.num_classes+1):
                self.tp_cls[cid]+=((pt==cid)&(pp==cid)).sum()
                self.fp_cls[cid]+=((pt!=cid)&(pp==cid)).sum() + (pred_cl[unP]==cid).sum()
                self.fn_cls[cid]+=((pt==cid)&(pp!=cid)).sum() + (labels[unT]==cid).sum()
            # ECE
            conf,argm = p["segmentation_mask"].max(0)
            cf = conf.cpu().numpy().ravel()
            cr = (argm==tm).cpu().numpy().ravel().astype(np.float32)
            self.pixels += cf.size
            inds = np.digitize(cf, self.bin_edges[1:-1], right=False)
            for b in range(self.nbins):
                m = inds==b
                if m.any():
                    self.bin_conf[b]+=cf[m].sum()
                    self.bin_corr[b]+=cr[m].sum()
                    self.bin_count[b]+=m.sum()
            # cache for PQ & viz if test
            if not self.train:
                self.preds.append({"ws": ws, "cls": pred_cl})
                self.targets.append({"gt_mask": tm_np, "gt_labels": labels})
            self.n+=1

    def _save_visualization(
        self,
        image: Union[np.ndarray, torch.Tensor],
        gt_mask: np.ndarray,
        watershed_mask: np.ndarray,
        true_gaussian: np.ndarray,
        pred_gaussian: np.ndarray,
        filename_prefix: str = "output",
        output_sufix: str = "output",
        dataset: str = "pannuke"
    ) -> None:
        import matplotlib.pyplot as plt
        # set colors/legend
        if dataset=="consep":
            class_colors=[[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0]]
            legend=[
                plt.Line2D([0],[0],marker='o',color='w',label='Misc',markerfacecolor='r',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Inflam',markerfacecolor='g',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Epit',markerfacecolor='b',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Spindle',markerfacecolor='y',markersize=10)
            ]
        elif dataset=="ki67":
            class_colors=[[0,0,0],[255,0,0],[0,255,0],[0,0,255]]
            legend=[
                plt.Line2D([0],[0],marker='o',color='w',label='Class1',markerfacecolor='r',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Class2',markerfacecolor='g',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Class3',markerfacecolor='b',markersize=10)
            ]
        else:
            class_colors=[[0,0,0],[255,0,0],[0,255,0],[255,255,0],[255,255,255],[0,0,255]]
            legend=[
                plt.Line2D([0],[0],marker='o',color='w',label='Neop',markerfacecolor='r',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Inflam',markerfacecolor='g',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Connect',markerfacecolor='y',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Necro',markerfacecolor='w',markersize=10),
                plt.Line2D([0],[0],marker='o',color='w',label='Epit',markerfacecolor='b',markersize=10)
            ]
        # image to numpy
        if isinstance(image, torch.Tensor):
            image = image.permute(2,0,1).cpu().numpy()
        image = np.clip(image,0,1)
        # color GT & pred
        gt_col = np.zeros((*gt_mask.shape,3),dtype=np.uint8)
        pr_col = np.zeros((*watershed_mask.shape,3),dtype=np.uint8)
        for cls in range(len(class_colors)):
            gt_col[gt_mask==cls] = class_colors[cls]
            pr_col[watershed_mask==cls] = class_colors[cls]
        fig,axs=plt.subplots(2,3,figsize=(15,10))
        axs[0,0].imshow(image); axs[0,0].set_title("Original")
        axs[0,1].imshow(gt_col); axs[0,1].set_title("GT"); axs[0,1].legend(handles=legend,loc="upper right")
        axs[0,2].imshow(pr_col); axs[0,2].set_title("Watershed")
        axs[1,0].imshow(true_gaussian,cmap="jet"); axs[1,0].set_title("True Gauss")
        axs[1,1].imshow(pred_gaussian,cmap="jet"); axs[1,1].set_title("Pred Gauss")
        plt.suptitle(f"{filename_prefix} | {output_sufix}")
        plt.tight_layout()
        save_dir="./final_outputs"; os.makedirs(save_dir,exist_ok=True)
        save_path=osp.join(save_dir,f"{filename_prefix}_{output_sufix}.png")
        plt.savefig(save_path,dpi=150); plt.close()
        print(f"Visualization saved to {save_path}")

    def compute(self) -> Dict[str,Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vals = torch.tensor([
            self.dice_sum, self.mse_sum, self.hn_sum, self.n,
            self.tp, self.fp, self.fn, self.tp_tn, self.fp_fn
        ], device=device, dtype=torch.float64)
        vals = self._all_reduce(vals).cpu().numpy().tolist()
        dice_sum,mse_sum,hn_sum,n_tot,tp,fp,fn,tp_tn,fp_fn = vals
        f1d,precd,recd,accd = cell_detection_scores(
            np.zeros(int(tp)),np.zeros(int(tp)),np.zeros(int(fn)),np.zeros(int(fp))
        )
        if self.pixels>0:
            mc = self.bin_conf/np.maximum(self.bin_count,1)
            mr = self.bin_corr/np.maximum(self.bin_count,1)
            ece = (np.abs(mr-mc)*(self.bin_count/self.pixels)).sum()
        else:
            ece=0.0
        out={
            "dice":dice_sum/max(n_tot,1),
            "mse":mse_sum/max(n_tot,1),
            "hn_dice":hn_sum/max(n_tot,1),
            "ece":float(ece),
            "detection":{"f1":f1d,"prec":precd,"rec":recd,"acc":accd}
        }
        for cid in range(1,self.num_classes+1):
            tp_c,fp_c,fn_c = self.tp_cls[cid],self.fp_cls[cid],self.fn_cls[cid]
            prec=tp_c/(tp_c+fp_c+1e-6); rec=tp_c/(tp_c+fn_c+1e-6)
            f1=2*tp_c/(2*tp_c+fp_c+fn_c+1e-6)
            out[self.class_names[cid-1]]={"f1":float(f1),"prec":float(prec),"rec":float(rec)}
        if not self.train:
            self.synchronize_between_processes()
            for idx, (pred_dict, tgt_dict) in enumerate(zip(self.preds, self.targets)):
                # pred_dict must contain everything needed:
                #   - raw image
                #   - gt_mask
                #   - watershed_mask (pred_dict["ws"])
                #   - true_gaussian, pred_gaussian, etc.
                # You may need to cache those in update() if you didn’t already.
                self._save_visualization(
                    image            = pred_dict["image"],
                    gt_mask          = tgt_dict["gt_mask"],
                    watershed_mask   = pred_dict["ws"],
                    true_gaussian    = tgt_dict["true_gaussian"],
                    pred_gaussian    = pred_dict["centroid_gaussian"],
                    filename_prefix  = f"sample_{idx}",
                    output_sufix     = self.output_suffix,
                    dataset          = self.dataset_name,  # if you stored it
                )
            all_ws  = [d["ws"]  for d in self.preds]
            all_cls = [d["cls"] for d in self.preds]
            all_gt  = [d["gt_mask"] for d in self.targets]
            all_gt_lbl=[d["gt_labels"] for d in self.targets]
            bPQ,mPQ = compute_bPQ_and_mPQ(
                remap_label_and_class_map(all_ws,all_cls),
                remap_label_and_class_map(all_gt,all_gt_lbl)
            )
            out["PQ"]=float(bPQ); out["mPQ"]=float(mPQ)
        return out


################################################################################
# ------------------------ MultiTaskEvaluationMetric --------------------------#
################################################################################

class MultiTaskEvaluationMetric(BaseCellMetric):
    """Streaming multi-task metric (Dice, MSE, ECE, detection + PQ/visualization)."""
    def __init__(
        self,
        num_classes:int,
        class_names:Optional[List[str]]=None,
        dataset_name:str="pannuke",
        max_pair_distance:float=12.0,
        train:bool=True,
        th:float=0.1,
        output_sufix:Optional[str]=None
    ) -> None:
        super().__init__(num_classes, class_names)
        self.max_pair_distance=max_pair_distance
        self.train=train
        self.th=th
        self.datset_name=dataset_name
        if output_sufix is None:
            self.output_sufix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        else:
            self.output_sufix = output_sufix


def build_metric(num_classes:int, class_names:Optional[List[str]]=None) -> BaseCellMetric:
    """Factory, same API as before."""
    return MultiTaskEvaluationMetric(num_classes, class_names)
