#! /usr/bin/env python
# coding=utf-8

import sys
sys.path.append("./op")
sys.path.append("./loss")

import tensorflow as tf
from loss.loss_helper_base import LossHelperBase
from op.DFTopK import DFTopK

EPS = tf.constant(1e-8, dtype=tf.float32)
NEG_INF = tf.constant(-1e9, dtype=tf.float32)

def tf_get_shape_at_axis(input, axis): #
    # static size
    if input is None:
        return 0
    return input.get_shape().as_list()[axis]

def get_set_value_by_topk_prob_and_label(topk_prob, label, top_k): #label [batchsize,N]
    t = topk_prob  # [batch_size,N]
    value = tf.reduce_sum(t*label, axis=-1)  # [batch_size]
    optimal_value, _ = tf.nn.top_k(label, k=top_k)
    set_value_sample_wise = value / tf.reduce_sum(optimal_value, axis=-1)
    return tf.reduce_mean(set_value_sample_wise)


def topk_label(scores, k):
    """
    scores: shape [B, N]
    k: int
    return: mask of top-k scores, shape [B, N]
    """
    topk_indices = tf.nn.top_k(scores, k=k).indices  # 获取 top-k 索引
    N = tf.shape(scores)[1]  # 特征维度
    one_hot_mask = tf.one_hot(topk_indices, depth=N, dtype=tf.float32)  # one-hot 编码
    mask = tf.reduce_sum(one_hot_mask, axis=1)  # 合并为 mask
    return mask

class ThresholdTopKLossv2(LossHelperBase):
    """
    参数：
      top_k: gt的数量
      support_m: 模型供给到下阶段的数量
      model_name, 需要计算loss的model instance的名字（regist model时填的name）
    """
    def loss_graph(self):
        top_k = self.conf.top_k
        tau = self.conf.tau

        grouped_logits = self.model_outputs[self.conf.model_name]
        grouped_labels = self.label_infos['grouped_labels']
        
        topk_labels = topk_label(grouped_labels, top_k)
        topk_logtis = DFTopK(grouped_logits, top_k)
        loss = -tf.reduce_mean(topk_labels * tf.log(topk_logtis+1e-6), axis=-1)
        loss_sample_wise = tf.reduce_mean(loss, axis=-1)
        loss_sample_wise = loss_sample_wise * tf.cast(self.label_infos['count'] > top_k, tf.float32)
        
        if hasattr(self.conf, 'sample_weight'):
            loss = tf.reduce_mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = tf.reduce_mean(loss_sample_wise)
        self.loss_output_dict[self.name] = loss

    def _init_check(self):
        assert hasattr(self.conf, 'top_k'), '{ERROR} %s missing argument joint_recall_k' % self.name
        assert hasattr(self.conf, 'model_name'), '{ERROR} %s missing argument model_name' % self.name
        assert 'grouped_labels' in self.label_infos, '{ERROR} %s missing label: grouped_labels' % self.name

class ThresholdTopKJointLoss(LossHelperBase):
    """
    参数：
      top_k: gt的数量
      support_m: 模型供给到下阶段的数量
      model_name, 需要计算loss的model instance的名字（regist model时填的name）
    """
    def loss_graph(self):
        tau = self.conf.tau

        grouped_prerank_logits = self.model_outputs[self.conf.prerank_model_name]
        grouped_recall_logits = self.model_outputs[self.conf.recall_model_name]
        grouped_labels = self.label_infos['grouped_labels']

        topk_labels = topk_label(grouped_labels, self.conf.gt_num)
        topk_prerank_logtis = DFTopK(grouped_prerank_logits, self.conf.joint_prerank_k)
        topk_recall_logits = DFTopK(grouped_recall_logits, self.conf.joint_recall_k)
        joint_logits = topk_prerank_logtis * topk_recall_logits
        loss = -tf.reduce_mean(topk_labels * tf.log(joint_logits+1e-6), axis=-1)
        loss_sample_wise = tf.reduce_mean(loss, axis=-1)
        loss_sample_wise = loss_sample_wise * tf.cast(self.label_infos['count'] > self.conf.joint_prerank_k, tf.float32)
        
        if hasattr(self.conf, 'sample_weight'):
            loss = tf.reduce_mean(loss_sample_wise * self.conf.sample_weight)
        else:
            loss = tf.reduce_mean(loss_sample_wise)
        self.loss_output_dict[self.name] = loss
        if hasattr(self.conf, 'is_main_loss') and self.conf.is_main_loss:
            prerank_acc = get_set_value_by_topk_prob_and_label(topk_prerank_logtis,
                                                    self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            self.loss_output_dict['prerank_r/r*'] = prerank_acc
            recall_acc = get_set_value_by_topk_prob_and_label(
                topk_recall_logits,
                self.label_infos['grouped_labels'], top_k=self.conf.gt_num)
            self.loss_output_dict['prerank_r/r*'] = recall_acc

    def _init_check(self):
        assert hasattr(self.conf, 'top_k'), '{ERROR} %s missing argument top_k' % self.name
        # assert hasattr(self.conf, 'model_name'), '{ERROR} %s missing argument model_name' % self.name
        assert 'grouped_labels' in self.label_infos, '{ERROR} %s missing label: grouped_labels' % self.name

class LCRON(LossHelperBase):
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, use_name_as_scope=True, is_debug=False,
                 is_train=True):
        super(LCRON, self).__init__(name=name, label_infos=label_infos, model_outputs=model_outputs,
                                    loss_conf=loss_conf,
                                    logger=logger, use_name_as_scope=use_name_as_scope, is_debug=is_debug,
                                    is_train=is_train)

        grouped_labels = model_outputs['grouped_labels']
        self.label_infos['grouped_labels'] = grouped_labels
        model_outputs[self.conf.prerank_model_name] = model_outputs['grouped_prerank_logits']
        model_outputs[self.conf.recall_model_name] = model_outputs['grouped_recall_logits']
        
        self.l_relax_helper_prerank = ThresholdTopKLossv2(name=self.name + '/L_relax_prerank',
                                                    label_infos=label_infos,
                                                    model_outputs={self.conf.prerank_model_name: model_outputs[
                                                        self.conf.prerank_model_name]},
                                                    loss_conf=type("", (), {"model_name": self.conf.prerank_model_name,
                                                                            "top_k": self.conf.gt_num,
                                                                            "tau":self.conf.tau,
                                                                            "support_m": self.conf.gt_num}),
                                                    logger=logger,
                                                    use_name_as_scope=use_name_as_scope,
                                                    is_debug=is_debug,
                                                    is_train=is_train)

        self.l_relax_helper_recall = ThresholdTopKLossv2(name=self.name + '/L_relax_recall',
                                                label_infos=label_infos,
                                                model_outputs=model_outputs,
                                                loss_conf=type("", (), {"model_name": self.conf.recall_model_name,
                                                                        "top_k": self.conf.gt_num,
                                                                        "tau":self.conf.tau,
                                                                        "support_m": self.conf.gt_num}),
                                                logger=logger,
                                                use_name_as_scope=use_name_as_scope,
                                                is_debug=is_debug,
                                                is_train=is_train)

        conf = type("", (), {"prerank_model_name": self.conf.prerank_model_name,
                                "recall_model_name": self.conf.recall_model_name,
                                "joint_recall_k": self.conf.joint_recall_k,
                                "joint_prerank_k": self.conf.joint_prerank_k,
                                "tau":self.conf.tau,
                                "gt_num": self.conf.gt_num,
                                "top_k": self.conf.top_k,
                                "is_main_loss": True})

        print(" joint conf joint_recall_k is: {}".format(self.conf.joint_recall_k))
        print(" joint conf joint_prerank_k is: {}".format(self.conf.joint_prerank_k))
        
        self.joint_loss_helper = ThresholdTopKJointLoss(name=self.name + '/L_joint',
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
            weight_shape = [1]  # for Kai
        else:
            weight_shape = ()  # for KLearn

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
