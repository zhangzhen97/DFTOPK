import torch
from abc import ABC, abstractmethod
class LossHelperBase(torch.nn.Module):
    def __init__(self, name, label_infos, model_outputs, loss_conf, logger, use_name_as_scope=True, is_debug=False,
                 is_train=True):
        super(LossHelperBase, self).__init__()

        self.name = name
        self.conf = loss_conf
        self.label_infos = label_infos
        self.model_outputs = model_outputs
        self.use_name_as_scope = use_name_as_scope
        self.logger = logger
        self.is_debug = is_debug
        self.is_train = is_train
        self.loss_output_dict = {}
        self._init_check()

    def _usage(self):
        pass

    def _init_check(self):
        pass

    def __calc_loss(self):
        self.loss_graph()

    def get_loss(self, name=None):
        self.__calc_loss()

        if name is None and len(self.loss_output_dict) == 1:
            for key in self.loss_output_dict.keys():
                return self.loss_output_dict[key]
        else:
            return self.loss_output_dict[name]

    @abstractmethod
    def loss_graph(self):
        pass

    def reset_model_output(self, model_output_dict):
        self.model_output_dict = model_output_dict

    def reset_name(self, name):
        self.name = name

    def reset_conf(self, loss_conf):
        self.conf = loss_conf

    def clear(self):
        self.loss_output_dict.clear()
        # self.__loss_is_ready = False

    @staticmethod
    def get_default_conf(model_name):
        return type("", (), {"model_name": model_name})

