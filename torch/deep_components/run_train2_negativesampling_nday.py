import itertools
import sys
import os
import argparse
import time
import torch
from torch.utils.data import DataLoader

from models_negativesampling import DSSM, DIN
import models
from dataset2 import Rank_Train_All_BY_RANK_Dataset
from utils import load_pkl
from deep_components.loss.two_stage.different_sorting_loss import compute_lcron_metrics_negative_sample, LcronLossModel
from deep_components.loss.two_stage.topk_loss import compute_cascade_topk_metrics, TopKLossModel_joint

import logging
import numpy as np

LOG_FORMAT = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

### GLOBAL
# rank_pos(10) + rank_neg(10) + coarse_neg(10) + prerank_neg(10)
max_num = 40
joint_loss_conf = type("", (), {"prerank_model_name": 'joint/prerank_model',
                                "recall_model_name": 'joint/recall_model',
                                "joint_recall_k": 30,
                                "joint_prerank_k": 20,
                                "gt_num": 10,
                                "global_size": max_num})

# conf for ARF:
prerank_arf_loss_conf = type("", (), {
    "model_name": 'prerank_model',
    "top_k": 10,
    "support_m": 20,
    "gt_num": 10,
    "global_size": max_num,
})

retrival_arf_loss_conf = type("", (), {
    "model_name": 'retrival_model',
    "top_k": 10,
    "support_m": 30,
    "gt_num": 10,
    "global_size": max_num,
})


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
    parser.add_argument('--root_path', type=str, default=".", help='root path to data, checkpoints and logs')
    parser.add_argument('--tau', type=float, default=1, help='tau.')
    parser.add_argument('--loss_type', type=str, default="fsltr", help='method type.')
    parser.add_argument('--sort_type', type=str, default='neural_sort', help='sort type')
    parser.add_argument('--sample_num', type=int, default=0, help="negative sampling num")
    return parser.parse_args()


def print_model(model, desc, num_limit=10):
    kv_ls = [(name, param) for name, param in model.named_parameters()]
    kv_ls.sort(key=lambda x: x[0])
    if num_limit > 0:
        for name, param in kv_ls[:num_limit] + kv_ls[-num_limit:]:
            print("print_model[%s].Parameter %s = %s" % (desc, name, param))
    else:
        for name, param in kv_ls:
            print("print_model[%s].Parameter %s = %s" % (desc, name, param))


def track_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"【{tag}】Current Memory: {allocated:.2f} MB | max: {max_allocated:.2f} MB")

def select_samples_all_batches(
    rank_pos: torch.Tensor,
    rank_neg: torch.Tensor,
    coarse_neg: torch.Tensor,
    prerank_neg: torch.Tensor,
    n: int
) -> torch.Tensor:

    bs, p, k = rank_pos.shape
    device = rank_pos.device
    total_per_batch = 4 * p 
    batch_indices = torch.arange(bs, device=device).unsqueeze(0).expand(bs, bs)
    batch_indices = batch_indices[~torch.eye(bs, dtype=torch.bool, device=device)].view(bs, bs-1)
    batch_indices_exp = batch_indices.unsqueeze(-1).expand(-1, -1, total_per_batch)           # (bs, bs-1, 4p)
    item_indices_exp = torch.arange(total_per_batch, device=device).view(1, 1, -1).expand(bs, bs - 1, -1)  # (bs, bs-1, 4p)
    flat_batch_indices = batch_indices_exp.reshape(bs, -1)  # (bs, (bs-1)*4p)
    flat_item_indices = item_indices_exp.reshape(bs, -1)    # (bs, (bs-1)*4p)
    total_candidates = (bs - 1) * total_per_batch
    sample_positions = torch.multinomial(
        torch.ones(bs, total_candidates, device=device), n, replacement=False
    )  
    selected_batch_ids = flat_batch_indices.gather(1, sample_positions)  # (bs, n)
    selected_item_ids = flat_item_indices.gather(1, sample_positions)    # (bs, n)
    sampled_b = selected_batch_ids.reshape(-1)  # (bs * n,)
    sampled_i = selected_item_ids.reshape(-1)   # (bs * n,)
    stacked = torch.cat([rank_pos, rank_neg, coarse_neg, prerank_neg], dim=1)
    sampled = stacked[sampled_b, sampled_i]     # (bs * n, dim)
    sampled = sampled.view(bs, n, k)          # (bs, n, dim)

    return sampled


if __name__ == '__main__':

    def run_train():
        t1 = time.time()
        print("set default device to GPU")

        args = parse_args()
        for k, v in vars(args).items():
            print(f"{k}:{v}")

        # negative sampling num
        max_num = 40 + args.sample_num
        joint_loss_conf.global_size = max_num
        prerank_arf_loss_conf.global_size = max_num
        retrival_arf_loss_conf.global_size = max_num

        # prepare data
        root_path = args.root_path
        
        prefix = root_path + "/data/"
        realshow_prefix = os.path.join(prefix, "all_stage")
        path_to_train_csv_lst = []
        with open("./deep_components/file_1st.txt", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                tmp_csv_path = os.path.join(realshow_prefix, line.strip() + '.feather')
                path_to_train_csv_lst.append(tmp_csv_path)

        num_of_train_csv = len(path_to_train_csv_lst)
        print("training files:%s" % path_to_train_csv_lst)
        print(f"number of train_csv: {num_of_train_csv}")
        for idx, filepath in enumerate(path_to_train_csv_lst):
            print(f"{idx}: {filepath}")

        seq_prefix = os.path.join(prefix, "seq_effective_50_dict")
        path_to_train_seq_pkl_lst = []
        with open("./deep_components/file_1st.txt", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                tmp_seq_pkl_path = os.path.join(seq_prefix, line.strip() + '.pkl')
                path_to_train_seq_pkl_lst.append(tmp_seq_pkl_path)

        print("training seq files:")
        for idx, filepath in enumerate(path_to_train_seq_pkl_lst):
            print(f"{idx}: {filepath}")

        request_id_prefix = os.path.join(prefix, "request_id_dict")
        path_to_train_request_pkl_lst = []
        with open("./deep_components/file_1st.txt", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                tmp_request_pkl_path = os.path.join(request_id_prefix, line.strip() + ".pkl")
                path_to_train_request_pkl_lst.append(tmp_request_pkl_path)

        print("training request files")
        for idx, filepath in enumerate(path_to_train_request_pkl_lst):
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

        if args.sample_num > 0:
            prerank_model = DIN(
                args.emb_dim, args.seq_len,
                device, id_cnt_dict,
                nn_units=args.nn_units
            ).to(device)
            retrival_model = DSSM(
                args.emb_dim, args.seq_len,
                device, id_cnt_dict,
                nn_units=args.nn_units).to(device)
        else:
            prerank_model = models.DIN(
                args.emb_dim, args.seq_len,
                device, id_cnt_dict,
                nn_units=args.nn_units
            ).to(device)
            retrival_model = models.DSSM(
                args.emb_dim, args.seq_len,
                device, id_cnt_dict,
                nn_units=args.nn_units).to(device)

        prerank_optimizer = torch.optim.Adam(prerank_model.parameters(), lr=args.lr)
        retrival_optimizer = torch.optim.Adam(retrival_model.parameters(), lr=args.lr)

        if args.loss_type.startswith("lcron"):
            loss_model = LcronLossModel(device)
            loss_optimizer = torch.optim.Adam(loss_model.parameters(), lr=args.lr)
        elif args.loss_type.startswith("cascade-topk"):
            loss_model = TopKLossModel_joint(device)
            loss_optimizer = torch.optim.Adam(loss_model.parameters(), lr=args.lr)
        else:
            loss_optimizer = None

        num_workers,rank_offset = 1,0
        # train each model with just one epoch. epcoh is used to check the variance of metrics.
        for epoch in [args.epochs]:
            retrival_model.train()
            prerank_model.train()
            if args.epochs > 1:
                prefix = "/checkpoints/E%s" % (epoch)
            else:
                prefix = "/checkpoints/"
            for n_day in range(num_of_train_csv):
                print("TRAIN. processing n_day:%s" % (n_day))
                train_dataset = Rank_Train_All_BY_RANK_Dataset(
                    path_to_train_csv_lst[n_day],
                    args.seq_len,
                    path_to_train_seq_pkl_lst[n_day],
                    path_to_train_request_pkl_lst[n_day],
                    rank_offset=rank_offset
                )
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    drop_last=True
                )

                print("{def} args.loss_type=%s" % args.loss_type)
                for iter_step, inputs in enumerate(train_loader):
                    # inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in inputs[:15]]

                    rank_pos, rank_neg, coarse_neg, prerank_neg = inputs[11:15]
                    rank_index_list = inputs[-4:]
                    mask_list = inputs[-8:-4]
                    # n = 40  # sample nums
                    if args.sample_num > 0:
                        all_lib_rank = torch.zeros(rank_index_list[0].shape[0], args.sample_num)
                        all_lib_mask = torch.ones(inputs[-8].shape[0],args.sample_num)
                        all_lib_neg_photo = select_samples_all_batches(rank_pos, rank_neg, coarse_neg, prerank_neg, args.sample_num)
                    
                    feature_input_list = inputs[:15]
                    if args.sample_num > 0:
                        feature_input_list.append(all_lib_neg_photo)
                        rank_index_list.append(all_lib_rank)
                        mask_list.append(all_lib_mask)

                    # print('mask_size',mask_list[-1].shape[1])

                    inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in feature_input_list]

                    if args.sample_num > 0:
                        inputs_LongTensor.append(args.sample_num)

                    prerank_logits_list = prerank_model.forward_all_by_rank(inputs_LongTensor)
                    retrival_logits_list = retrival_model.forward_all_by_rank(inputs_LongTensor)

                    for logit in prerank_logits_list:
                        logit.retain_grad()
                    for logit in retrival_logits_list:
                        logit.retain_grad()

                    if iter_step % 100 == 0:
                        track_memory("BACKWARD_MEM")
                    if args.loss_type.startswith("lcron"):
                        if len(args.loss_type.split("_")) > 1:
                            version = args.loss_type.split("_")[1]
                        else:
                            version = 'v0'
                        joint_loss_conf.version = version
                        outputs = compute_lcron_metrics_negative_sample(rank_index_list, mask_list, prerank_logits_list, retrival_logits_list, device,
                                                        max_num=max_num,
                                                        joint_loss_conf=joint_loss_conf,
                                                        logger=logger,
                                                        loss_model=loss_model,
                                                        sort=args.sort_type)
                        loss = outputs["total_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        loss_optimizer.zero_grad()
                        loss.backward()
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        loss_optimizer.step()
                        if iter_step % args.print_freq == 0:
                            print(f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")
                    elif args.loss_type.startswith("cascade-topk"):
                        if len(args.loss_type.split("_")) > 1:
                            version = args.loss_type.split("_")[1]
                        else:
                            version = ''
                        alpha = torch.tensor(5, device=device)
                        outputs = compute_cascade_topk_metrics([],prerank_logits_list, retrival_logits_list, device,
                                                        joint_loss_conf=joint_loss_conf,version = version, tau=args.tau, loss_model = loss_model, rank_index_list=rank_index_list , mask_list=mask_list, alpha = alpha)
                        loss = outputs["total_loss"]
                        prerank_optimizer.zero_grad()
                        retrival_optimizer.zero_grad()
                        loss_optimizer.zero_grad()
                        loss.backward()
                        # print("prerank_logits:",logits[0], logits.grad[0])
                        prerank_optimizer.step()
                        retrival_optimizer.step()
                        loss_optimizer.step()

                        if iter_step % args.print_freq == 0:
                            print(
                                f"State=Union. loss={args.loss_type} Day:{n_day}\t[Epoch/iter]:{epoch:>3}/{iter_step:<4}\tloss:{loss.detach().cpu().item():.4f} ")
            
                path_to_save_model = root_path + prefix + f"prerank_tau-{args.loss_type}-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}-SN{args.sample_num}_D{n_day}.pkl"
                torch.save(prerank_model.state_dict(), path_to_save_model)
                print("Saving DAILY prerank model to path_to_save_model:%s" % path_to_save_model)
                path_to_save_model = root_path + prefix + f"retrival_tau-{args.loss_type}-{args.tau}--bs-{args.batch_size}_lr-{args.lr}_{args.tag}-SN{args.sample_num}_D{n_day}.pkl"
                torch.save(retrival_model.state_dict(), path_to_save_model)
                print("Saving DAILY retrival model to path_to_save_model:%s" % path_to_save_model)

        t2 = time.time()
        print("time_used:%s" % (t2 - t1))

    run_train()
