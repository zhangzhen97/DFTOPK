import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_components.loss.loss_helper_base import LossHelperBase
from deep_components.loss.sorting import neuralsort, soft_sort
from deep_components.loss.two_stage.google_top_k import sparse_soft_topk_mask_dykstra
from deep_components.loss.two_stage.lapsum import log_soft_top_k, soft_top_k

EPS = 1e-8
NEG_INF = -1e9

def tensor_concat(logits_list, rank_index_list, mask_list, device):
    mask_tensor = torch.cat([mask.to(device) for mask in mask_list], dim=1)  
    logits_tensor = torch.cat([logits.to(device) for logits in logits_list], dim=1)  
    rank_index_tensor = torch.cat([rank_index.to(device) for rank_index in rank_index_list],
                                  dim=1)  
    mask_sum_per_pv = mask_tensor.sum(dim=1)
    return logits_tensor, rank_index_tensor, mask_tensor, None, mask_sum_per_pv

def dftopk(x, k, tau=1.0):
    x_k,_ = torch.kthvalue(x=-x, k=k, dim=1)    
    x_k_plus_1,_ = torch.kthvalue(x=-x, k=k+1, dim=1)
    threshold = ((x_k + x_k_plus_1) / 2.0).unsqueeze(1)
    logits = x+threshold
    if tau!=1:logits = logits / tau
    return logits


def topk_label(scores, k):
    """
    scores: shape [B, N]
    k: int
    return: mask of top-k scores, shape [B, N]
    """
    topk_indices = torch.topk(scores, k=k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.float32)
    mask.scatter_(dim=1, index=topk_indices, value=1)
    return mask

def topk_label_kthvalue(scores, k):
    """
    scores: shape [B, N]
    k: int
    return: mask of top-k scores, shape [B, N]
    """
    kth_values, _ = torch.kthvalue(scores, k=scores.size(1) - k + 1, dim=1)
    mask = (scores >= kth_values.unsqueeze(1)).float()
    extra = mask.sum(dim=1, keepdim=True) - k
    if extra.max() > 0:
        topk_indices = torch.topk(scores, k=k, dim=1).indices
        new_mask = torch.zeros_like(scores, dtype=torch.float32)
        new_mask.scatter_(1, topk_indices, 1)
        mask = new_mask
    return mask

class TopKLossModel_joint(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.prerank_weight = torch.nn.Parameter(torch.ones(1, requires_grad=True, device=device))
        self.recall_weight = torch.nn.Parameter(torch.ones(1, requires_grad=True, device=device))

    def forward(self, l_relax_prerank, l_relax_recall, l_joint):
        final_loss = (0.5 / torch.square(self.prerank_weight)) * l_relax_prerank + \
                     (0.5 / torch.square(self.recall_weight)) * l_relax_recall + \
                    l_joint + \
                    torch.log(self.prerank_weight*self.recall_weight)
        
        print("DEBUG_LcronLossModel[prerank_weight=%s, recall_weight=%s]" % (self.prerank_weight, self.recall_weight))
        return final_loss

def compute_cascade_topk_metrics(inputs, prerank_logits, retrival_logits, device, joint_loss_conf, version="", tau=1.0, loss_model=None, rank_index_list=None, mask_list=None, alpha=0):
    if inputs != []:
        rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
        mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]

    prerank_sorted_logits, sorted_rank_index_list, _, _, mask_sum_per_pv_list = tensor_concat(
        logits_list=prerank_logits,
        rank_index_list=rank_index_list,
        mask_list=mask_list,
        device=device)

    retrival_sorted_logits, _, _, _, _ = tensor_concat(
        logits_list = retrival_logits,
        rank_index_list = rank_index_list,
        mask_list = mask_list,
        device = device)

    count = mask_sum_per_pv_list
    count = count.to(device)
    
    top_k = joint_loss_conf.gt_num
    if version == "sigv0uwjointv3":
        labels = topk_label(sorted_rank_index_list.float(), top_k)
        labels_bottom_10 = topk_label(-sorted_rank_index_list.float(), 10)
        labels_bottom_20 = topk_label(-sorted_rank_index_list.float(), 20)
    elif version == "dftopkjoint":
        labels = topk_label_kthvalue(sorted_rank_index_list.float(), top_k)
    else:
        labels = topk_label(sorted_rank_index_list.float(), top_k)

    if version == "dftopkjoint":
        prerank_logits_topk = dftopk(prerank_sorted_logits, top_k, tau)
        retrival_logits_topk = dftopk(retrival_sorted_logits, top_k, tau)
        prerank_loss_sample_wise = F.binary_cross_entropy_with_logits(
        prerank_logits_topk, 
        labels,
        reduction='none'
        ).mean(dim=-1)
        recall_loss_sample_wise = F.binary_cross_entropy_with_logits(
            retrival_logits_topk, 
            labels,
            reduction='none'
        ).mean(dim=-1)
        recall_loss_sample_wise  = torch.mean(recall_loss_sample_wise)
        prerank_loss_sample_wise = torch.mean(prerank_loss_sample_wise)
        prerank_probs = torch.sigmoid(prerank_logits_topk)
        retrival_probs = torch.sigmoid(retrival_logits_topk)
        joint_probs = prerank_probs * retrival_probs 
        loss_fn = nn.BCELoss(reduction='none')
        joint_loss = loss_fn(joint_probs, labels).mean(dim=-1)
        joint_loss = torch.mean(joint_loss)
        total_loss = loss_model.forward(prerank_loss_sample_wise, recall_loss_sample_wise, joint_loss)
    elif version == "googlejoint":
        prerank_topk_prob = sparse_soft_topk_mask_dykstra(prerank_sorted_logits, top_k, l=1)
        recall_topk_prob = sparse_soft_topk_mask_dykstra(retrival_sorted_logits, top_k, l=1)
        prerank_loss_sample_wise = F.binary_cross_entropy_with_logits(
        prerank_topk_prob, 
        labels,
        reduction='none'
        ).mean(dim=-1)
        recall_loss_sample_wise = F.binary_cross_entropy_with_logits(
            recall_topk_prob, 
            labels,
            reduction='none'
        ).mean(dim=-1)
        recall_loss_sample_wise  = torch.mean(recall_loss_sample_wise)
        prerank_loss_sample_wise = torch.mean(prerank_loss_sample_wise)
        prerank_probs = torch.sigmoid(prerank_topk_prob)
        retrival_probs = torch.sigmoid(recall_topk_prob)
        joint_probs = prerank_probs * retrival_probs 
        loss_fn = nn.BCELoss(reduction='none')
        joint_loss = loss_fn(joint_probs, labels).mean(dim=-1)
        joint_loss = torch.mean(joint_loss)
        total_loss = loss_model.forward(prerank_loss_sample_wise, recall_loss_sample_wise, joint_loss)
    elif version == "lapsumjoint":
        topk_input = torch.full((prerank_sorted_logits.shape[0],), top_k, dtype=torch.float32, device=prerank_sorted_logits.device)
        prerank_probs = soft_top_k(prerank_sorted_logits, topk_input, alpha=alpha, descending=False)
        recall_probs  = soft_top_k(retrival_sorted_logits, topk_input, alpha=alpha, descending=False)
        prerank_loss_sample_wise = -torch.mean(torch.log(prerank_probs+1e-6)*labels+(1-labels)*torch.log(1-prerank_probs+1e-6))
        recall_loss_sample_wise = -torch.mean(torch.log(recall_probs+1e-6)*labels+(1-labels)*torch.log(1-recall_probs+1e-6))
        joint_probs = prerank_probs * recall_probs
        joint_probs = prerank_probs * recall_probs
        joint_loss = -torch.mean(torch.log(joint_probs+1e-6)*labels+(1-labels)*torch.log(1-joint_probs+1e-6))
        total_loss = loss_model.forward(prerank_loss_sample_wise, recall_loss_sample_wise, joint_loss) 
    outputs = {"total_loss": total_loss}
    return outputs
