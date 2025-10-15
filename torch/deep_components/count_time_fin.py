import sys
import os
import argparse
import time
import torch
from deep_components.loss.sorting import SortNet
from deep_components.loss.two_stage.topk_loss import dftopk
import torch.nn.functional as F
from deep_components.loss.two_stage.google_top_k import sparse_soft_topk_mask_dykstra
from deep_components.loss.two_stage.lapsum import soft_top_k, soft_top_k_autograd
import torch.nn as nn
from utils import load_pkl
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size.')
    parser.add_argument('--tau', type=float, default=1, help='tau.')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device.')
    parser.add_argument('--num', type=int, default=0, help="num of item")
    parser.add_argument('--k', type=int, default=0, help="topk")
    return parser.parse_args()

class DSSM(nn.Module):
  def __init__(self, emb_dim, seq_len, device, nn_units=128):
    super(DSSM, self).__init__()

    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.nn_units = nn_units

    # user
    self.uid_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.vid_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.cate_two_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    self.up_type_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # context
    self.wday_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.hour_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.min_emb = nn.Embedding(
      num_embeddings=100,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # encoder
    self.user_encoder = nn.Sequential(
      nn.Linear(emb_dim * 13, self.nn_units),
      nn.ReLU(),
      nn.Linear(self.nn_units, self.nn_units//2),
      nn.ReLU(),
      nn.Linear(self.nn_units//2, self.nn_units//4)
    )

    self.photo_encoder = nn.Sequential(
      nn.Linear(emb_dim * 8, self.nn_units),
      nn.ReLU(),
      nn.Linear(self.nn_units, self.nn_units//2),
      nn.ReLU(),
      nn.Linear(self.nn_units//2, self.nn_units//4)
    )

  def forward_all_by_rank(self, inputs):
    request_wday, request_hour, request_min, \
      uid, did, gender, age, province, \
      seq_arr, seq_len, \
      rank_pos_photos= inputs

    # Context embeddings
    req_wda_emb = self.wday_emb(request_wday)
    req_hou_emb = self.hour_emb(request_hour)
    req_min_emb = self.min_emb(request_min)

    # User embeddings
    uid_emb = self.uid_emb(uid)
    did_emb = self.did_emb(did)
    gen_emb = self.gender_emb(gender)
    age_emb = self.age_emb(age)
    pro_emb = self.province_emb(province)

    # Behavior embeddings
    seq_len = seq_len.float().unsqueeze(-1)  # b*1
    vid_seq_emb = self.vid_emb(seq_arr[:, :, 0])  # b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:, :, 1])
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:, :, 2])
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:, :, 3])
    up_seq_emb = self.up_type_emb(seq_arr[:, :, 4])

    seq_emb = torch.cat([vid_seq_emb, aid_seq_emb, cate_two_seq_emb, cate_one_seq_emb, up_seq_emb],
                        dim=2)  # b*seq_len*5d
    seq_sum = torch.sum(seq_emb, dim=1)  # b*5d
    seq_emb_mean = seq_sum / seq_len  # b*5d

    # User input
    u_input = torch.cat([
      req_wda_emb, req_hou_emb, req_min_emb,
      uid_emb, did_emb, gen_emb, age_emb, pro_emb,
      seq_emb_mean
    ], dim=1)  # b*(8+5)d
    u_out = self.user_encoder(u_input)  # b*32

    def compute_photo_logits(photo_inputs):
      # Extract photo embeddings
      vid_emb = self.vid_emb(photo_inputs[:, :, 0])  # b*p*d
      aid_emb = self.aid_emb(photo_inputs[:, :, 1])
      cate_two_emb = self.cate_two_emb(photo_inputs[:, :, 2])
      cate_one_emb = self.cate_one_emb(photo_inputs[:, :, 3])
      up_emb = self.up_type_emb(photo_inputs[:, :, 4])
      up_wda_emb = self.wday_emb(photo_inputs[:, :, 5])
      up_hou_emb = self.hour_emb(photo_inputs[:, :, 6])
      up_min_emb = self.min_emb(photo_inputs[:, :, 7])

      # Photo input
      p_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
                           up_wda_emb, up_hou_emb, up_min_emb], dim=2)  # b*p*8d
      p_out = self.photo_encoder(p_input)  # b*p*32

      # Compute logits
      logits = torch.bmm(p_out, u_out.unsqueeze(dim=-1)).squeeze()  # b*p
      return logits

    # Compute logits for each type of photo inputs
    rank_pos_logits = compute_photo_logits(rank_pos_photos)  # b*10

    # Return all logits
    return rank_pos_logits if len(rank_pos_logits.shape)!=1 else rank_pos_logits.unsqueeze(0)

def compute_lcron_metrics_count_time(inputs,labels,device,k,max_num, tau=50, sort="neural_sort"):
    sort_op_config = SortNet.get_default_config(sort)
    if sort == 'neural_sort' or sort == "soft_sort":
        sort_op_config['tau'] = tau
    sort_op_config['device'] = device
    sort_op_config['size'] = max_num
    sort_net = SortNet(sort_op=sort, reverse=False, config=sort_op_config)

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
    with torch.no_grad():
        label_matrix = sort_net_label.forward(labels.float().detach())
    permutation_matrix = sort_net.forward(inputs)
    top_k = k
    support_m = max_num
    detach_permutation_matrix = permutation_matrix.detach()
    up_target_permutation_matrix = torch.sum(permutation_matrix[:, :support_m, :], dim=-2)
    raw_sum_permutation_matrix = torch.sum(detach_permutation_matrix, dim=-2)
    up_target_permutation_matrix = up_target_permutation_matrix / (raw_sum_permutation_matrix + 1e-6)
    up_target_label_matrix = torch.sum(label_matrix[:, :top_k, :], dim=-2)
    up_loss_sample_wise = torch.mean(-torch.log(up_target_permutation_matrix + 1e-6) * up_target_label_matrix,dim=-1)
    down_target_permutation_matrix = torch.sum(permutation_matrix[:, support_m:, :], dim=-2)
    down_target_permutation_matrix = down_target_permutation_matrix / (raw_sum_permutation_matrix + 1e-6)
    down_target_label_matrix = torch.sum(label_matrix[:, top_k:, :], dim=-2)
    down_loss_sample_wise = torch.mean(-torch.log(down_target_permutation_matrix + 1e-6) * down_target_label_matrix, dim=-1)
    loss = up_loss_sample_wise + down_loss_sample_wise
    return torch.mean(loss)

def compute_cascade_count_time(inputs,labels,device,k, version="",tau=1.0,alpha=0):
    if version == "dftopkjoint":
        inputs_logits_topk = dftopk(inputs, k, tau)
        inputs_loss_sample_wise = F.binary_cross_entropy_with_logits(
        inputs_logits_topk, 
        labels,
        reduction='none'
        ).mean(dim=-1)
        loss = torch.mean(inputs_loss_sample_wise)
    elif version == "googlejoint":
        inputs_topk_prob = sparse_soft_topk_mask_dykstra(inputs, k, l=1)
        prerank_loss_sample_wise = F.binary_cross_entropy_with_logits(
        inputs_topk_prob, 
        labels,
        reduction='none'
        ).mean(dim=-1)
        prerank_loss_sample_wise = torch.mean(prerank_loss_sample_wise)
        loss = torch.mean(prerank_loss_sample_wise)
    elif version == "lapsumjoint":
        topk_input = torch.full((inputs.shape[0],), k, dtype=torch.float32, device=device)
        prerank_probs = soft_top_k_autograd(inputs, topk_input, alpha=alpha, descending=False)
        loss = -torch.mean(torch.log(prerank_probs+1e-6)*labels+(1-labels)*torch.log(1-prerank_probs+1e-6))

    return loss

def create_label(batch_size, n, k):
    rand_indices = torch.rand(batch_size, n).argsort(dim=1)
    mask = torch.zeros(batch_size, n, dtype=torch.bool)
    mask.scatter_(1, rand_indices[:, :k], True)
    return mask.float()

Methods = [("cascade-topk_dftopkjoint",""),("cascade-topk_googlejoint",""),("cascade-topk_lapsumjoint",""),("lcron","soft_sort"),("lcron","neural_sort"),("lcron","diff_sort")]

if __name__ == '__main__':
    def run_train():
        args = parse_args()
        if torch.cuda.is_available() and args.cuda >= 0:
            device = torch.device(f"cuda:{args.cuda}")
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
        labels = create_label(args.batch_size, args.num, args.k).to(device)
        batchsize = args.batch_size
        model = DSSM(
          8, 50,
          device,
          nn_units=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for method in Methods:
            loss_type = method[0]
            sort_type = method[1]
            for _ in range(10): 
                inputs_LongTensor = [
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize, 50, 5), dtype=torch.int64).to(device),
                torch.full((batchsize,), 50, dtype=torch.int64).to(device), 
                torch.randint(0, 100, (batchsize, args.num, 8), dtype=torch.int64).to(device)
                ]  
                inputs = model.forward_all_by_rank(inputs_LongTensor)
                if loss_type.startswith("lcron"):
                    loss = compute_lcron_metrics_count_time(inputs, labels, device=device, k=args.k, max_num=args.num, tau=args.tau, sort=sort_type)
                elif loss_type.startswith("cascade-topk"):
                    version = loss_type.split("_")[1]
                    alpha = torch.tensor(5, device=device, dtype=torch.float32)
                    loss = compute_cascade_count_time(inputs, labels, device=device, k=args.k, version=version, tau=args.tau, alpha=alpha)
                loss.backward()
                optimizer.zero_grad()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            times = [] 
            for i in range(100):
                if device.type == 'cuda':
                    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    starter.record()
                else: # CPU
                    start_time = time.perf_counter()
                inputs_LongTensor = [
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),
                torch.randint(0, 100, (batchsize,), dtype=torch.int64).to(device),

                torch.randint(0, 100, (batchsize, 50, 5), dtype=torch.int64).to(device),
                torch.full((batchsize,), 50, dtype=torch.int64).to(device),  
                torch.randint(0, 100, (batchsize, args.num, 8), dtype=torch.int64).to(device)
                ]  
                inputs = model.forward_all_by_rank(inputs_LongTensor)
                # print('inputs',inputs)
                if loss_type.startswith("lcron"):
                    loss = compute_lcron_metrics_count_time(inputs, labels, device=device, k=args.k, max_num=args.num, tau=args.tau, sort=sort_type)
                elif loss_type.startswith("cascade-topk"):
                    version = loss_type.split("_")[1]
                    alpha = torch.tensor(5, device=device, dtype=torch.float32)
                    loss = compute_cascade_count_time(inputs, labels, device=device, k=args.k, version=version, tau=args.tau, alpha=alpha)
                loss.backward()
                optimizer.zero_grad()
                # -----------------------------

                if device.type == 'cuda':
                    ender.record()
                    torch.cuda.synchronize(device)
                    curr_time = starter.elapsed_time(ender) 
                else: # CPU
                    end_time = time.perf_counter()
                    curr_time = (end_time - start_time) * 1000 
                
                times.append(curr_time)
            avg_time = sum(times[10:]) / len(times[10:]) if len(times) > 10 else sum(times) / len(times)
            print(f"loss:{loss_type}-{sort_type}  Batchsize={args.batch_size}  N={args.num}  K={args.k}  avg_time:{avg_time} ms")

    run_train()