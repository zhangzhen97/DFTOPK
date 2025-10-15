import sys
import os
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

from models_negativesampling import DSSM, DIN
import models
from metrics import evaluate, evaluate_join, evaluate_join_negativesampling, evaluate_negativesampling
from dataset2 import Rank_Train_All_BY_RANK_Dataset
from utils import load_pkl

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args(): 
  parser = argparse.ArgumentParser()

  parser.add_argument('--epochs', type=int, default=1, help='epochs.')
  parser.add_argument('--batch_size', type=int, default=1024, help='train batch size.')
  parser.add_argument('--infer_realshow_batch_size', type=int, default=1024, help='inference batch size.')
  parser.add_argument('--infer_recall_batch_size', type=int, default=1024, help='inference batch size.')
  parser.add_argument('--nn_units', type=int, default=128, help='nn units.')
  parser.add_argument('--emb_dim', type=int, default=8, help='embedding dimension.')
  parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
  parser.add_argument('--seq_len', type=int, default=3, help='length of behaivor sequence')
  parser.add_argument('--cuda', type=int, default=0, help='cuda device.')
  parser.add_argument('--print_freq', type=int, default=200, help='frequency of print.')
  parser.add_argument('--tag', type=str, default="1st", help='exp tag.')
  parser.add_argument('--tau', type=float, default=1, help='tau.')
  parser.add_argument('--root_path', type=str, default=".", help='root path to data, checkpoints and logs')
  parser.add_argument('--loss_type', type=str, default="fsltr", help='method type.')
  parser.add_argument('--sample_num', type=int, default=0, help="negative sampling num")
  return parser.parse_args()

def print_model(model, desc, num_limit=10):
    kv_ls = [(name, param) for name,param in model.named_parameters()]
    kv_ls.sort(key=lambda  x: x[0])
    if num_limit>0:
        for name, param in kv_ls[:num_limit] + kv_ls[-num_limit:]:
            print("print_model[%s].Parameter %s = %s"%(desc, name, param))
    else:
        for name, param in kv_ls:
            print("print_model[%s].Parameter %s = %s"%(desc, name, param))

if __name__ == '__main__':

    def run_eval():

        args = parse_args()
        for k, v in vars(args).items():
            print(f"{k}:{v}")

        # prepare data
        root_path = args.root_path
        prefix = root_path + "/data/"

        eval_day_list = ['2024-02-04']
        print("eval_day_list:%s"%eval_day_list)
        realshow_prefix = os.path.join(prefix, "all_stage")
        path_to_eval_csv_lst = []
        for line in eval_day_list:
            tmp_csv_path = os.path.join(realshow_prefix, line.strip() + '.feather')
            path_to_eval_csv_lst.append(tmp_csv_path)

        num_of_train_csv = len(path_to_eval_csv_lst)
        print("testing files:")
        print(f"number of train_csv: {num_of_train_csv}")
        for idx, filepath in enumerate(path_to_eval_csv_lst):
            print(f"{idx}: {filepath}")

        seq_prefix = os.path.join(prefix, "seq_effective_50_dict")
        path_to_eval_seq_pkl_lst = []
        for line in eval_day_list:
            tmp_seq_pkl_path = os.path.join(seq_prefix, line.strip() + '.pkl')
            path_to_eval_seq_pkl_lst.append(tmp_seq_pkl_path)

        print("testing seq files:")
        for idx, filepath in enumerate(path_to_eval_seq_pkl_lst):
            print(f"{idx}: {filepath}")

        request_id_prefix = os.path.join(prefix, "request_id_dict")
        path_to_eval_request_pkl_lst = []
        for line in eval_day_list:
            tmp_request_pkl_path = os.path.join(request_id_prefix, line.strip() + ".pkl")
            path_to_eval_request_pkl_lst.append(tmp_request_pkl_path)

        print("testing request files")
        for idx, filepath in enumerate(path_to_eval_request_pkl_lst):
            print(f"{idx}: {filepath}")

        others_prefix = os.path.join(prefix, "others")
        path_to_id_cnt_pkl = os.path.join(others_prefix, "id_cnt.pkl")
        print(f"path_to_id_cnt_pkl: {path_to_id_cnt_pkl}")

        id_cnt_dict = load_pkl(path_to_id_cnt_pkl)
        for k, v in id_cnt_dict.items():
            print(f"{k}:{v}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        # seed = 1024
        # np.random.seed(seed)

        for epoch in [args.epochs]:
            if args.epochs > 1:
                prefix = "/checkpoints/E%s" % (epoch)
            else:
                prefix = "/checkpoints/"

            print("TEST_STEP. load models:epoch=%s"%epoch)

            prerank_model = DIN(
                args.emb_dim, args.seq_len,
                device, id_cnt_dict,
                nn_units=args.nn_units
            ).to(device)

            retrival_model = DSSM(
                args.emb_dim, args.seq_len,
                device, id_cnt_dict,
                nn_units=args.nn_units
            ).to(device)           

            path_to_save_model = root_path + prefix + f"prerank_tau-{args.loss_type}-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}-SN{args.sample_num}.pkl"
            state_dict = torch.load(path_to_save_model)
            prerank_model.load_state_dict(state_dict)
            print("Loading prerank model to path_to_save_model:%s" % path_to_save_model)
            path_to_save_model = root_path + prefix + f"retrival_tau-{args.loss_type}-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}-SN{args.sample_num}.pkl"
            state_dict = torch.load(path_to_save_model)
            retrival_model.load_state_dict(state_dict)
            print("Loading retrival model to path_to_save_model:%s" % path_to_save_model)

            print("TEST_STEP. switch to eval mode (for BatchNorm etc.)")
            prerank_model.eval()
            retrival_model.eval()

            is_debug = True
            print("{def} args.loss_type=%s" % args.loss_type)
            print("{def} is_debug=%s" % is_debug)

            n_day = 0
            dataset = Rank_Train_All_BY_RANK_Dataset(
                path_to_eval_csv_lst[n_day],
                args.seq_len,
                path_to_eval_seq_pkl_lst[n_day],
                path_to_eval_request_pkl_lst[n_day])
            loader = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=True
            )
            # evaluate_join_negativesampling(prerank_model, retrival_model, loader, device, 160, desc="%s\t%s" % (args.loss_type, "joint"))
            evaluate_join_negativesampling(prerank_model, retrival_model, loader, device, 160, desc="%s\t%s" % (args.loss_type, "joint"))

            for model, name in [[prerank_model, "prerank"], [retrival_model, "retrival"]]:
                dataset = Rank_Train_All_BY_RANK_Dataset(
                    path_to_eval_csv_lst[n_day],
                    args.seq_len,
                    path_to_eval_seq_pkl_lst[n_day],
                    path_to_eval_request_pkl_lst[n_day])

                loader = DataLoader(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=True
                )
                evaluate_negativesampling(model, loader, device, 160, is_debug=False, desc="%s\t%s"%(args.loss_type, name))

    run_eval()
