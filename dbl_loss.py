# dbl_loss_tsd
import torch
import torch.nn as nn
import torch.nn.functional as F

# helper IoU
def box_area(box):
    return ((box[...,2]-box[...,0]).clamp(min=0) * (box[...,3]-box[...,1]).clamp(min=0))

def box_iou(pred, target):
    # pred [N,4], target [N,4]
    lt = torch.max(pred[:,:2], target[:,:2])
    rb = torch.min(pred[:,2:], target[:,2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,0]*wh[:,1]
    area1 = box_area(pred)
    area2 = box_area(target)
    union = area1 + area2 - inter + 1e-12
    iou = inter / union
    return iou, inter, union

class DBL_TSD(nn.Module):
    """
    Dynamic Balanced Loss with TSD-inspired scale-aware box term.
    Usage: produce preds dict and targets dict compatible with ultralytics callback.
    """
    def __init__(self, cls_w=1.0, box_w=5.0, iou_w=2.0, scale_alpha=1.0, trunc_thresh=0.4):
        super().__init__()
        self.cls_w = cls_w
        self.box_w = box_w
        self.iou_w = iou_w
        self.scale_alpha = scale_alpha
        self.trunc_thresh = trunc_thresh
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: dict with keys 'cls' [P,C], 'box' [P,4], 'iou' [P] (or [P,1])
        targets: dict with same shapes (soft labels for cls ok)
        Note: this expects flattened predictions for a batch (depends on trainer).
        """
        # ------ classification (balanced) ------
        cls_pred = preds['cls']      # [P,C]
        cls_tgt  = targets['cls']    # [P,C] soft labels
        cls_loss = self.cls_loss_fn(cls_pred, cls_tgt)  # [P,C]
        # class frequency weight (per-batch)
        freq = cls_tgt.sum(dim=0) + 1e-6   # [C]
        inv = 1.0 / freq
        inv = inv / inv.sum() * freq.numel()
        cls_loss = (cls_loss * inv.unsqueeze(0)).mean()

        # ------ box regression with scale-aware weight (TSD idea) ------
        box_pred = preds['box']   # [P,4]
        box_tgt  = targets['box'] # [P,4]
        pos_mask = (targets.get('obj', None) is not None and targets['obj']>0) \
                   if 'obj' in targets else ( (box_tgt.abs().sum(dim=1)>0) )
        if pos_mask is None:
            pos_mask = torch.zeros(box_pred.size(0), dtype=torch.bool, device=box_pred.device)

        box_loss = torch.tensor(0., device=box_pred.device)
        iou_loss = torch.tensor(0., device=box_pred.device)
        if pos_mask.any():
            ppos = box_pred[pos_mask]
            tpos = box_tgt[pos_mask]
            # smooth L1 (Huber)
            diff = torch.abs(ppos - tpos)
            l1 = torch.where(diff < 1.0, 0.5 * diff**2, diff - 0.5).mean(dim=1)  # per-obj
            # scale factor: smaller gt area -> larger factor (TSD-inspired)
            gt_area = box_area(tpos)  # per-obj
            median_area = torch.median(gt_area) if gt_area.numel()>0 else 1.0
            scale_factor = 1.0 + self.scale_alpha * (median_area / (gt_area + 1e-6))
            scale_factor = scale_factor.clamp(max=10.0)  # avoid explosion
            box_loss = (l1 * scale_factor).mean()

            # IoU loss with truncation emphasis for small objects
            iou_val, _, _ = box_iou(ppos, tpos)
            # truncated penalty: emphasize when prediction IoU < trunc_thresh
            trunc_mask = (iou_val < self.trunc_thresh).float()
            iou_pen = ((1.0 - iou_val) ** 2) * (1.0 + trunc_mask * 2.0)   # more penalty if below threshold
            iou_loss = iou_pen.mean()

        # total
        total = self.cls_w * cls_loss + self.box_w * box_loss + self.iou_w * iou_loss
        # return scalar loss and breakdown
        info = {
            "cls_loss": float(cls_loss.detach().cpu()) if isinstance(cls_loss, torch.Tensor) else cls_loss,
            "box_loss": float(box_loss.detach().cpu()) if isinstance(box_loss, torch.Tensor) else 0.0,
            "iou_loss": float(iou_loss.detach().cpu()) if isinstance(iou_loss, torch.Tensor) else 0.0
        }
        return total, info
