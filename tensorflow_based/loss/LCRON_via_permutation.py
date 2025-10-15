#! /usr/bin/env python
# coding=utf-8

import sys

sys.path.append("./op")
sys.path.append("./loss")
import tensorflow as tf
from loss.loss_helper_base import LossHelperBase
from op.differentiale_sorting import SortNet

EPS = tf.constant(1e-8, dtype=tf.float32)
NEG_INF = tf.constant(-1e9, dtype=tf.float32)

def tf_get_shape_at_axis(input, axis):
    # static size
    if input is None:
        return 0
    return input.get_shape().as_list()[axis]

def get_set_value_by_permutation_matrix_and_label(permutation_matrix, label, top_k):
    t = tf.squeeze(tf.matmul(permutation_matrix, tf.expand_dims(label, axis=-1)), axis=[2])  # [batch_size,N]
    value = tf.reduce_sum(t[:, :top_k], axis=-1)  # [batch_size]
    optimal_value, _ = tf.nn.top_k(label, k=top_k)
    set_value_sample_wise = value / tf.reduce_sum(optimal_value, axis=-1)
    return tf.reduce_mean(set_value_sample_wise)

class JointLoss(LossHelperBase):
    """
    LCRON的joint loss
    参数：
      conf.joint_recall_k, recall出口quota
      conf.joint_prerank_k, prerank出口quota
      conf.gt_num, gt数量
      conf.prerank_model_name,
      conf.recall_model_name,
      [optional] conf.min_size_for_joint_train (原代码中固定设为9)
    """
    # 要求label越大越好
    # 此时label为该ad的ecpm或rank_index
    # 传入的若是rank index应该做预处理，让rank index越大越好。
    def loss_graph(self):
        pre_rank_permutation_matrix = self.model_outputs[self.conf.prerank_model_name]
        recall_permutation_matrix = self.model_outputs[self.conf.recall_model_name]
        all_label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = tf.expand_dims(mask_all, axis=1) * tf.expand_dims(mask_all, axis=2)
        pre_rank_permutation_matrix_mask = pre_rank_permutation_matrix * s2_mask
        recall_permutation_matrix_mask = recall_permutation_matrix * s2_mask
        all_label_matrix_mask = all_label_matrix * s2_mask
        joint_recall_k = self.conf.joint_recall_k
        target_recall_permutation_matrix = tf.reduce_mean(recall_permutation_matrix_mask[:, :joint_recall_k, :],
                                                          axis=-2)
        joint_prerank_k = self.conf.joint_prerank_k
        target_pre_rank_permutation_matrix = tf.reduce_mean(pre_rank_permutation_matrix_mask[:, :joint_prerank_k, :],
                                                            axis=-2)
        target_joint_permutation_matrix = target_recall_permutation_matrix * target_pre_rank_permutation_matrix
        target_all_label_matrix = tf.reduce_sum(all_label_matrix_mask[:, :self.conf.gt_num, :], axis=-2)
        joint_loss = tf.reduce_mean(-tf.log(target_joint_permutation_matrix + 1e-6) * target_all_label_matrix, axis=-1)
        if hasattr(self.conf, 'min_size_for_joint_train'):
            joint_loss = joint_loss * tf.cast(self.label_infos['count'] > self.conf.min_size_for_joint_train, tf.float32)
        joint_loss = tf.reduce_mean(joint_loss)
        self.loss_output_dict[self.name] = joint_loss
        if hasattr(self.conf, 'is_main_loss') and self.conf.is_main_loss:
            prerank_acc = get_set_value_by_permutation_matrix_and_label(
                self.model_outputs[self.conf.prerank_model_name],
                self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            recall_acc = get_set_value_by_permutation_matrix_and_label(
                self.model_outputs[self.conf.recall_model_name],
                self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            self.loss_output_dict['prerank_r/r*'] = prerank_acc
            self.loss_output_dict['recall_r/r*'] = recall_acc

    def _init_check(self):
        assert hasattr(self.conf, 'joint_recall_k'), '{ERROR} %s missing argument joint_recall_k' % self.name
        assert hasattr(self.conf, 'joint_prerank_k'), '{ERROR} %s missing argument joint_prerank_k' % self.name
        assert hasattr(self.conf, 'gt_num'), '{ERROR} %s missing argument gt_num' % self.name
        assert hasattr(self.conf, 'prerank_model_name'), '{ERROR} %s missing argument prerank_model_name' % self.name
        assert hasattr(self.conf, 'recall_model_name'), '{ERROR} %s missing argument recall_model_name' % self.name
        assert 'label_permutation_matrix' in self.label_infos, '{ERROR} %s missing label: label_permutation_matrix' % self.name

class LRelaxLoss(LossHelperBase):
    """
    参数：
      top_k: gt的数量
      support_m: 模型供给到下阶段的数量
      model_name, 需要计算loss的model instance的名字（regist model时填的name）
    """
    def loss_graph(self):
        top_k = self.conf.top_k
        support_m = self.conf.support_m
        permutation_matrix = self.model_outputs[self.conf.model_name]
        label_matrix = self.label_infos['label_permutation_matrix']
        mask_all = self.label_infos['label_mask']
        s2_mask = tf.expand_dims(mask_all, axis=1) * tf.expand_dims(mask_all, axis=2)
        permutation_matrix = permutation_matrix * s2_mask
        label_matrix = label_matrix * s2_mask
        target_permutation_matrix = tf.reduce_mean(permutation_matrix[:, :support_m, :], axis=-2)
        target_label_matrix = tf.reduce_sum(label_matrix[:, :top_k, :], axis=-2)
        if self.is_debug:
            tf.summary.histogram("target_permutation_matrix", target_permutation_matrix)
            tf.summary.histogram("target_label_matrix", target_label_matrix)
        loss_sample_wise = tf.reduce_mean(-tf.log(target_permutation_matrix + 1e-6) * target_label_matrix * self.label_infos['label_mask'],
            axis=-1)
        loss_sample_wise = loss_sample_wise * tf.cast(self.label_infos['count'] > support_m, tf.float32)
        if hasattr(self.conf, 'sample_weight'):
            loss = tf.reduce_mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = tf.reduce_mean(loss_sample_wise)
        self.loss_output_dict[self.name] = loss
        if hasattr(self.conf, 'is_main_loss') and self.conf.is_main_loss:
            acc = get_set_value_by_permutation_matrix_and_label(self.model_outputs[self.conf.model_name]['logits_permutation_matrix'],
                                                    self.label_infos['grouped_labels'], top_k=self.conf.top_k)
            self.loss_output_dict['r/r*'] = acc


    def _init_check(self):
        assert hasattr(self.conf, 'top_k'), '{ERROR} %s missing argument joint_recall_k' % self.name
        assert hasattr(self.conf, 'support_m'), '{ERROR} %s missing argument support_m' % self.name
        assert hasattr(self.conf, 'model_name'), '{ERROR} %s missing argument model_name' % self.name
        assert 'label_permutation_matrix' in self.label_infos, '{ERROR} %s missing label: label_permutation_matrix' % self.name


class LCRON(LossHelperBase):
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, sort_op='neural_sort', use_name_as_scope=True, is_debug=False,
                 is_train=True):
        super(LCRON, self).__init__(name=name, label_infos=label_infos, model_outputs=model_outputs,
                                    loss_conf=loss_conf,
                                    logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                    is_train=is_train)
        sort_op_config = SortNet.get_default_config(sort_op)
        sort_op_config['tau'] = 1.0
        sort_net = SortNet(sort_op=sort_op, reverse=False, config=sort_op_config)
        label_sort_net = SortNet(sort_op=sort_op, reverse=False, config={'tau': 0.1})
        grouped_labels = model_outputs['grouped_labels']
        grouped_prerank_logits = model_outputs['grouped_prerank_logits']
        grouped_recall_logits = model_outputs['grouped_recall_logits']
        self.label_infos['grouped_labels'] = grouped_labels
        self.label_infos['label_permutation_matrix'] = label_sort_net.forward(grouped_labels)
        model_outputs[self.conf.prerank_model_name] = sort_net.forward(grouped_prerank_logits)
        model_outputs[self.conf.recall_model_name] = sort_net.forward(grouped_recall_logits)

        self.l_relax_helper_prerank = LRelaxLoss(name=self.name + '/L_relax_prerank',
                                                    label_infos=label_infos,
                                                    model_outputs={self.conf.prerank_model_name: model_outputs[
                                                        self.conf.prerank_model_name]},
                                                    loss_conf=type("", (), {"model_name": self.conf.prerank_model_name,
                                                                            "top_k": self.conf.gt_num,
                                                                            "support_m": self.conf.gt_num}),
                                                    logger=logger,
                                                    use_name_as_scope=use_name_as_scope,
                                                    is_debug=is_debug,
                                                    is_train=is_train)
        
        self.l_relax_helper_recall = LRelaxLoss(name=self.name + '/L_relax_recall',
                                                label_infos=label_infos,
                                                model_outputs=model_outputs,
                                                loss_conf=type("", (), {"model_name": self.conf.recall_model_name,
                                                                        "top_k": self.conf.gt_num,
                                                                        "support_m": self.conf.gt_num}),
                                                logger=logger,
                                                use_name_as_scope=use_name_as_scope,
                                                is_debug=is_debug,
                                                is_train=is_train)

        conf = type("", (), {"prerank_model_name": self.conf.prerank_model_name,
                                "recall_model_name": self.conf.recall_model_name,
                                "joint_recall_k": self.conf.joint_recall_k, # 召回送粗排的quota
                                "joint_prerank_k": self.conf.joint_prerank_k, # 粗排送精排的quota
                                "gt_num": self.conf.gt_num, # ground-truth size
                                "is_main_loss": True})

        self.joint_loss_helper = JointLoss(name=self.name + '/L_joint',
                                            label_infos=label_infos,
                                            model_outputs=model_outputs,
                                            loss_conf=conf,
                                            logger=logger,
                                            use_name_as_scope=use_name_as_scope,
                                            is_debug=is_debug,
                                            is_train=is_train)

    def _init_check(self):
        assert hasattr(self.conf, 'joint_recall_k'), '{ERROR} %s missing argument joint_recall_k' % self.name
        assert hasattr(self.conf, 'joint_prerank_k'), '{ERROR} %s missing argument joint_prerank_k' % self.name
        assert hasattr(self.conf, 'gt_num'), '{ERROR} %s missing argument gt_num' % self.name
        assert hasattr(self.conf, 'prerank_model_name'), '{ERROR} %s missing argument prerank_model_name' % self.name
        assert hasattr(self.conf, 'recall_model_name'), '{ERROR} %s missing argument recall_model_name' % self.name

    def loss_graph(self):
        if not hasattr(self.conf, 'set_weight_shape_1'):
            weight_shape = [1]
        elif self.conf.set_weight_shape_1:
            weight_shape = [1]
        else:
            weight_shape = ()

        prerank_local_weight = tf.get_variable(name='prerank_local_weight', shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.ones_initializer, trainable=True)
        recall_local_weight = tf.get_variable(name='recall_local_weight', shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.ones_initializer, trainable=True)
        joint_weight = tf.get_variable(name='joint_weight', shape=weight_shape, dtype=tf.float32,
                                    initializer=tf.ones_initializer, trainable=True)

        l_relax_prerank = self.l_relax_helper_prerank.get_loss(self.l_relax_helper_prerank.name)
        l_relax_recall = self.l_relax_helper_recall.get_loss(self.l_relax_helper_recall.name)
        l_joint = self.joint_loss_helper.get_loss(self.joint_loss_helper.name)

        final_loss = (0.5 / tf.square(prerank_local_weight)) * l_relax_prerank \
                        + (0.5 / tf.square(recall_local_weight)) * l_relax_recall \
                        + (0.5 / tf.square(joint_weight)) * l_joint \
                        + tf.math.log(prerank_local_weight) \
                        + tf.math.log(recall_local_weight) \
                        + tf.math.log(joint_weight)
        self.loss_output_dict['l_joint'] = l_joint
        self.loss_output_dict['l_relax_recall'] = l_relax_recall
        self.loss_output_dict['l_relax_prerank'] = l_relax_prerank
        self.loss_output_dict[self.name] = tf.squeeze(final_loss)
        self.loss_output_dict['prerank_r/r*'] = self.joint_loss_helper.loss_output_dict['prerank_r/r*']
        self.loss_output_dict['recall_r/r*'] = self.joint_loss_helper.loss_output_dict['recall_r/r*']