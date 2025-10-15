import torch
import torch.nn.functional as F
from deep_components.loss.loss_helper_base import LossHelperBase
from deep_components.loss.sorting import neuralsort, soft_sort, SortNet

def get_shape_at_axis(tensor, axis):
    if tensor is None:
        return 0
    try:
        return tensor.shape[axis]
    except IndexError:
        return 0

def tensor_concat(logits_list, rank_index_list, mask_list, device):
    mask_tensor = torch.cat([mask.to(device) for mask in mask_list], dim=1)  # shape=[batch_size, total_ad_num]
    logits_tensor = torch.cat([logits.to(device) for logits in logits_list], dim=1)  # shape=[batch_size, total_ad_num]
    rank_index_tensor = torch.cat([rank_index.to(device) for rank_index in rank_index_list],
                                  dim=1)  # shape=[batch_size, total_ad_num]
    mask_sum_per_pv = mask_tensor.sum(dim=1)
    return logits_tensor, rank_index_tensor, mask_tensor, None, mask_sum_per_pv


def sequence_mask(count, padding_to_len):
    range_tensor = torch.arange(padding_to_len, device=count.device).unsqueeze(0)
    range_tensor = range_tensor.expand(count.size(0), padding_to_len)
    mask = (count.unsqueeze(-1) > range_tensor).float()
    return mask

def get_set_value_by_permutation_matrix_and_label(permutation_matrix, label, top_k):
    t = torch.matmul(permutation_matrix, label.unsqueeze(-1)).squeeze(-1)  # [batch_size, N]
    value = torch.sum(t[:, :top_k], dim=-1)  # [batch_size]
    optimal_value, _ = torch.topk(label, k=top_k, dim=-1)
    set_value_sample_wise = value / torch.sum(optimal_value, dim=-1)
    return torch.mean(set_value_sample_wise)

def compute_lcron_metrics_negative_sample(rank_index_list, mask_list, prerank_logits, retrival_logits, device, loss_model, max_num, joint_loss_conf, logger, tau=50, sort="neural_sort"):
    # rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
    # mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]
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
    label_mask = sequence_mask(count, max_num)

    sort_op_config = SortNet.get_default_config(sort)
    if sort == 'neural_sort' or sort == "soft_sort":
        sort_op_config['tau'] = tau
    # print('tau', tau)
    sort_op_config['device'] = device
    sort_op_config['size'] = max_num
    sort_net = SortNet(sort_op=sort, reverse=False, config=sort_op_config)
    # print("sort_net.size",sort_net.net.size)

    if sort == "neural_sort":
        sort_op_config_label = SortNet.get_default_config(sort)
        sort_op_config_label['tau'] = 0.0001
    elif sort == "soft_sort":
        sort_op_config_label = SortNet.get_default_config(sort)
        sort_op_config_label['tau'] = 0.0001
    elif sort == "diff_sort":
        sort_op_config_label = SortNet.get_default_config(sort)
        sort_op_config_label['steepness'] = 1000

    sort_op_config_label['device'] = device
    sort_op_config_label['size'] = max_num
    sort_net_label = SortNet(sort_op=sort, reverse=False, config=sort_op_config_label)
    # print('diffshape',sorted_rank_index_list.shape)
    with torch.no_grad():
        label_permutation_matrix = sort_net_label.forward(sorted_rank_index_list.float().detach())
        # label_permutation_matrix = (label_permutation_matrix > 0.01).float()
    
    grouped_labels = label_permutation_matrix
    label_infos = {
        "label_permutation_matrix": label_permutation_matrix,
        "label_mask": label_mask,
        "grouped_labels": grouped_labels,
        "count": count
    }

    label_permutation_matrix = (label_permutation_matrix > 0.01).float()
    prerank_logits_matrix = sort_net.forward(prerank_sorted_logits)
    retrival_matrix = sort_net.forward(retrival_sorted_logits)
    model_outputs_dict = {
        "joint/prerank_model": {
            "logits_permutation_matrix": prerank_logits_matrix,
            "logits": prerank_sorted_logits
        },
        "joint/recall_model": {
            "logits_permutation_matrix": retrival_matrix,
            "logits": retrival_sorted_logits
        }
    }
    loss_instance = LCRON(name='joint/cascade_model', label_infos=label_infos,
                          model_outputs=model_outputs_dict,
                          loss_conf=joint_loss_conf,
                          logger=logger,
                          use_name_as_scope=True,
                          device=device,
                          loss_model=loss_model,
                          is_debug=True,
                          is_train=True)

    total_loss = loss_instance.get_loss('joint/cascade_model')

    print("DEBUG_LCRON_LOSS. l_relax_recall=%s\tl_relax_prerank=%s\tl_joint=%s" % (
        loss_instance.get_loss('l_relax_recall').detach().cpu().numpy(),
        loss_instance.get_loss('l_relax_prerank').detach().cpu().numpy(),
        loss_instance.get_loss('l_joint').detach().cpu().numpy()))

    #
    outputs = {"total_loss": total_loss}
    return outputs

class JointUltraLoss(LossHelperBase):
    def loss_graph(self):
        pre_rank_permutation_matrix = self.model_outputs[self.conf.prerank_model_name]['logits_permutation_matrix']
        recall_permutation_matrix = self.model_outputs[self.conf.recall_model_name]['logits_permutation_matrix']
        all_label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = mask_all.unsqueeze(1) * mask_all.unsqueeze(2)
        pre_rank_permutation_matrix_mask = pre_rank_permutation_matrix * s2_mask
        recall_permutation_matrix_mask = recall_permutation_matrix * s2_mask
        detach_pre_rank_permutation_matrix_mask = pre_rank_permutation_matrix_mask.detach()
        detach_recall_permutation_matrix_mask = recall_permutation_matrix_mask.detach()
        all_label_matrix_mask = all_label_matrix * s2_mask
        joint_recall_k = self.conf.joint_recall_k
        joint_prerank_k = self.conf.joint_prerank_k
        up_target_recall_permutation_matrix = torch.sum(recall_permutation_matrix_mask[:, :joint_recall_k, :], dim=-2)
        all_detach_target_recall_permutation_matrix = torch.sum(detach_recall_permutation_matrix_mask, dim=-2)
        up_target_recall_permutation_matrix = up_target_recall_permutation_matrix / (
                    all_detach_target_recall_permutation_matrix + 1e-6)
        up_target_pre_rank_permutation_matrix = torch.sum(pre_rank_permutation_matrix_mask[:, :joint_prerank_k, :],
                                                          dim=-2)
        all_detach_target_pre_rank_permutation_matrix = torch.sum(detach_pre_rank_permutation_matrix_mask, dim=-2)
        up_target_pre_rank_permutation_matrix = up_target_pre_rank_permutation_matrix / (
                    all_detach_target_pre_rank_permutation_matrix + 1e-6)
        up_target_joint_permutation_matrix = up_target_recall_permutation_matrix * up_target_pre_rank_permutation_matrix
        up_target_all_label_matrix = torch.sum(all_label_matrix_mask[:, :self.conf.gt_num, :], dim=-2)
        up_joint_loss = torch.mean(-torch.log(up_target_joint_permutation_matrix + 1e-6) * up_target_all_label_matrix,
                                   dim=-1)
        
        down_target_recall_permutation_matrix = torch.sum(recall_permutation_matrix_mask[:, joint_recall_k:, :], dim=-2)
        down_target_recall_permutation_matrix = down_target_recall_permutation_matrix/(all_detach_target_recall_permutation_matrix+1e-6)
        down_target_recall_permutation_matrix = 1 -  down_target_recall_permutation_matrix
        
        down_target_pre_rank_permutation_matrix = torch.sum(pre_rank_permutation_matrix_mask[:, joint_prerank_k:, :], dim=-2)
        down_target_pre_rank_permutation_matrix = down_target_pre_rank_permutation_matrix/(all_detach_target_pre_rank_permutation_matrix+1e-6)
        down_target_pre_rank_permutation_matrix = 1 -  down_target_pre_rank_permutation_matrix

        down_target_joint_permutation_matrix = down_target_recall_permutation_matrix * down_target_pre_rank_permutation_matrix
        down_target_all_label_matrix = torch.clamp(1 - up_target_all_label_matrix, min=0, max=1)
        down_joint_loss = torch.mean(-torch.log((1-down_target_joint_permutation_matrix) + 1e-6) * down_target_all_label_matrix, dim=-1)
        
        joint_loss = up_joint_loss + down_joint_loss
        joint_loss = torch.mean(joint_loss)
        self.loss_output_dict[self.name] = joint_loss

class LsingleLoss(LossHelperBase):
    def loss_graph(self):
        top_k = self.conf.top_k
        support_m = self.conf.support_m
        permutation_matrix = self.model_outputs[self.conf.model_name]['logits_permutation_matrix']
        label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = mask_all.unsqueeze(1) * mask_all.unsqueeze(2)
        permutation_matrix = permutation_matrix * s2_mask
        label_matrix = label_matrix * s2_mask
        detach_permutation_matrix = permutation_matrix.detach()
        up_target_permutation_matrix = torch.sum(permutation_matrix[:, :support_m, :], dim=-2)
        raw_sum_permutation_matrix = torch.sum(detach_permutation_matrix, dim=-2)
        up_target_permutation_matrix = up_target_permutation_matrix / (raw_sum_permutation_matrix + 1e-6)
        up_target_label_matrix = torch.sum(label_matrix[:, :top_k, :], dim=-2)
        up_loss_sample_wise = torch.mean(
            -torch.log(up_target_permutation_matrix + 1e-6) * up_target_label_matrix * self.label_infos['label_mask'],
            dim=-1)
        down_target_permutation_matrix = torch.sum(permutation_matrix[:, support_m:, :], dim=-2)
        down_target_permutation_matrix = down_target_permutation_matrix / (raw_sum_permutation_matrix + 1e-6)
        down_target_label_matrix = torch.sum(label_matrix[:, top_k:, :], dim=-2)
        down_loss_sample_wise = torch.mean(
            -torch.log(down_target_permutation_matrix + 1e-6) * down_target_label_matrix * self.label_infos[
                'label_mask'], dim=-1)
        loss_sample_wise = up_loss_sample_wise + down_loss_sample_wise
        loss_sample_wise = loss_sample_wise * (self.label_infos['count'] > support_m).float()
        if hasattr(self.conf, 'sample_weight'):
            loss = torch.mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = torch.mean(loss_sample_wise)
        self.loss_output_dict[self.name] = loss


class LCRON(LossHelperBase):
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, use_name_as_scope=True, is_debug=False,
                 is_train=True, device=None, loss_model=None):
        super(LCRON, self).__init__(name=name, label_infos=label_infos, model_outputs=model_outputs,
                                    loss_conf=loss_conf,
                                    logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                    is_train=is_train)
        self.device = device
        self.loss_model = loss_model        
        self.l_relax_helper_prerank = LsingleLoss(name=self.name + '/L_relax_prerank',
                                                    label_infos=label_infos,
                                                    model_outputs={self.conf.prerank_model_name: model_outputs[
                                                        self.conf.prerank_model_name]},
                                                    loss_conf=type("", (), {
                                                        "model_name": self.conf.prerank_model_name,
                                                        "top_k": self.conf.gt_num,
                                                        "support_m": self.conf.gt_num}),
                                                    logger=logger,
                                                    use_name_as_scope=use_name_as_scope,
                                                    is_debug=is_debug,
                                                    is_train=is_train)

        self.l_relax_helper_recall = LsingleLoss(name=self.name + '/L_relax_recall',
                                                    label_infos=label_infos,
                                                    model_outputs=model_outputs,
                                                    loss_conf=type("", (), {
                                                        "model_name": self.conf.recall_model_name,
                                                        "top_k": self.conf.gt_num,
                                                        "support_m": self.conf.gt_num}),
                                                    logger=logger,
                                                    use_name_as_scope=use_name_as_scope,
                                                    is_debug=is_debug,
                                                    is_train=is_train)

        conf = type("", (), {"prerank_model_name": self.conf.prerank_model_name,
                            "recall_model_name": self.conf.recall_model_name,
                            "joint_recall_k": self.conf.gt_num,
                            "joint_prerank_k": self.conf.gt_num,
                            "gt_num": self.conf.gt_num,
                            "global_size": self.conf.global_size})
        self.joint_loss_helper = JointUltraLoss(name=self.name + '/L_joint',
                                                label_infos=label_infos,
                                                model_outputs=model_outputs,
                                                loss_conf=conf,
                                                logger=logger,
                                                use_name_as_scope=use_name_as_scope,
                                                is_debug=is_debug,
                                                is_train=is_train)

    def loss_graph(self):
        l_relax_prerank = self.l_relax_helper_prerank.get_loss(self.l_relax_helper_prerank.name)
        l_relax_recall = self.l_relax_helper_recall.get_loss(self.l_relax_helper_recall.name)
        l_joint = self.joint_loss_helper.get_loss(self.joint_loss_helper.name)
        final_loss = self.loss_model.forward(l_relax_prerank, l_relax_recall, l_joint, self.conf.version)
        self.loss_output_dict['l_relax_recall'] = l_relax_recall
        self.loss_output_dict['l_relax_prerank'] = l_relax_prerank
        self.loss_output_dict['l_joint'] = l_joint
        self.loss_output_dict[self.name] = final_loss.squeeze()

class LcronLossModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.prerank_weight = torch.nn.Parameter(torch.ones(1, requires_grad=True, device=device))
        self.recall_weight = torch.nn.Parameter(torch.ones(1, requires_grad=True, device=device))

    def forward(self, l_relax_prerank, l_relax_recall, l_joint, version):
        final_loss = (0.5 / torch.square(self.prerank_weight)) * l_relax_prerank + \
                     (0.5 / torch.square(self.recall_weight)) * l_relax_recall + \
                    l_joint + \
                    torch.log(self.prerank_weight*self.recall_weight)
        print("DEBUG_LcronLossModel[prerank_weight=%s, recall_weight=%s]" % (self.prerank_weight, self.recall_weight))
        return final_loss
