#! /usr/bin/env python
# coding=utf-8

import os
if os.environ.get('tf_v1')=="T":
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
else:
    import tensorflow as tf
import sys
from abc import ABC, abstractmethod
sys.path.append("/loss")

def get_set_value_by_permutation_matrix_and_label(permutation_matrix, label, top_k):
    t = tf.squeeze(tf.matmul(permutation_matrix, tf.expand_dims(label, axis=-1)), axis=[2])  # [batch_size,N]
    value = tf.reduce_sum(t[:, :top_k], axis=-1)  # [batch_size]
    optimal_value, _ = tf.nn.top_k(label, k=top_k)
    set_value_sample_wise = value / tf.reduce_sum(optimal_value, axis=-1)
    return tf.reduce_mean(set_value_sample_wise)

class LossHelperBase:
    """
    用来处理复杂loss的基类，比如多输入、多输出的复杂loss
    输入model_output_dict, loss_conf, 输出 loss_output_dict
    简单loss没必要继承此类
    """
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, use_name_as_scope=True, is_debug=False, is_train=True):
        """
        name: loss 的名称
        label_infos: a dict of kvs, k is tensor name (str), v is tensor
        model_outputs: a dict of models outputs, k is model name (str), v is a model_instance.model_outputs_dict
        loss_conf: the conf of the loss
        logger: logger
        use_name_as_scope: if true, the loss will use self.name as a variable_scope
        is_debug: is debug
        is_train: is train
        """
        self.name = name
        self.conf = loss_conf
        self.label_infos = label_infos
        self.model_outputs = model_outputs
        self.use_name_as_scope = use_name_as_scope
        self.logger = logger
        self.is_debug = is_debug
        self.is_train = is_train
        self.loss_output_dict = {}
        self.__loss_is_ready = False
        self._init_check()

    def _usage(self):
        """
        可选项，可以记录&打印 loss 的用法
        """
        pass

    def _init_check(self):
        """
        可选项，可以为具体的loss设置一些检查项
        """
        pass

    def __calc_loss(self):
        if self.use_name_as_scope:
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self.loss_graph()
        else:
            self.loss_graph()
        self.__loss_is_ready = True

    def get_loss(self, name=None):
        if not self.__loss_is_ready:
            self.__calc_loss()
            self.__loss_is_ready = True
            assert len(self.loss_output_dict)>0, "loss output is empty, please checkout the loss_graph"
        if name is None and len(self.loss_output_dict)==1:
            for key in self.loss_output_dict.keys():
                return self.loss_output_dict[key]
        else:
            return self.loss_output_dict[name]

    @abstractmethod
    def loss_graph(self):
        """
        在这里实现loss的计算图
        """
        pass

    def reset_model_output(self, model_output_dict):
        self.model_output_dict = model_output_dict

    def reset_name(self, name):
        self.name = name

    def reset_conf(self, loss_conf):
        self.conf = loss_conf

    def clear(self):
        self.loss_output_dict.clear()
        self.__loss_is_ready=False

    @staticmethod
    def get_default_conf(model_name):
        loss_conf = type("", (), {"model_name":model_name})
        return loss_conf