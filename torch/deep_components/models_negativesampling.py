import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
  def __init__(
    self, 
    emb_dim,
    seq_len, 
    device, 
    id_cnt_dict,
    nn_units=128
  ):
    super(DIN, self).__init__()
    
    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.nn_units = nn_units

    #user
    self.uid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['user_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['device_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['gender'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['age'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['province'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #item
    # fix1:
    video_id = id_cnt_dict.get('photo_id')
    if video_id is None:
      video_id = id_cnt_dict.get('video_id')
    self.vid_emb = nn.Embedding(
      num_embeddings=video_id + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['author_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    # fix2:
    category_level_two = id_cnt_dict.get('category_level_two')
    if category_level_two is None:
      category_level_two = id_cnt_dict.get(' category_level_two')
    self.cate_two_emb = nn.Embedding(
      num_embeddings= category_level_two + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_one']+2,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    
    self.up_type_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['upload_type'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    #context
    self.wday_emb = nn.Embedding(
      num_embeddings= 7 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    
    self.hour_emb = nn.Embedding(
      num_embeddings= 24 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    
    self.min_emb = nn.Embedding(
      num_embeddings= 60 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.din_mlp = nn.Sequential(
      nn.Linear(emb_dim*10, 80),
      nn.ReLU(),
      nn.Linear(80, 1),
    )
    
    #encoder    
    self.mlp = nn.Sequential(
      nn.Linear(emb_dim*(8+13), self.nn_units),
      nn.ReLU(),
      nn.Linear(self.nn_units, self.nn_units),
      nn.ReLU(),
      nn.Linear(self.nn_units, self.nn_units//4),
      nn.ReLU(),
      nn.Linear(self.nn_units//4, 1)
    )

  def forward_all_by_rank(self, inputs):
    request_wday, request_hour, request_min, \
      uid, did, gender, age, province, \
      seq_arr, seq_mask, seq_len, \
      rank_pos_photos, rank_neg_photos, coarse_neg_photos, prerank_neg_photos, all_lib_neg_photos, n = inputs

    # Context embedding
    req_wda_emb = self.wday_emb(request_wday)  # b*d
    req_hou_emb = self.hour_emb(request_hour)  # b*d
    req_min_emb = self.min_emb(request_min)  # b*d

    # User embedding
    uid_emb = self.uid_emb(uid)  # b*d
    did_emb = self.did_emb(did)  # b*d
    gen_emb = self.gender_emb(gender)  # b*d
    age_emb = self.age_emb(age)  # b*d
    pro_emb = self.province_emb(province)  # b*d

    # Combine user-side embeddings
    user_side_emb = torch.cat([req_wda_emb, req_hou_emb, req_min_emb, uid_emb, did_emb, gen_emb, age_emb, pro_emb],
                              dim=1)  # b*8d
    user_side_emb_b10 = user_side_emb.unsqueeze(1).repeat([1, 10, 1])  # b*10*8d
    user_side_emb_bn =  user_side_emb.unsqueeze(1).repeat([1, n, 1])

    # Sequence embedding
    seq_mask_bool = torch.ne(seq_mask, 0)  # b*seq_len

    vid_seq_emb = self.vid_emb(seq_arr[:, :, 0])  # b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:, :, 1])  # b*seq_len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:, :, 2])  # b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:, :, 3])  # b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:, :, 4])  # b*seq_len*d
    # print("{def} forward_all_by_rank. self.vid_emb.weight.device=%s vid_seq_emb.device=%s" % (
    #       self.vid_emb.weight.device, vid_seq_emb.device))
    seq_emb = torch.cat([vid_seq_emb, aid_seq_emb, cate_two_seq_emb, cate_one_seq_emb, up_seq_emb],
                        dim=2)  # b*seq_len*5d
    seq_emb_b10 = seq_emb.repeat_interleave(10, dim=0)  # b10*seq_len*5d
    seq_emb_bn = seq_emb.repeat_interleave(n, dim=0)

    seq_mask_bool_b10 = seq_mask_bool.repeat_interleave(10, dim=0).unsqueeze(dim=1)  # b10*1*seq_len
    # print("{def} forward_all_by_rank. seq_emb.device=%s"%(seq_emb.device))
    seq_mask_bool_bn = seq_mask_bool.repeat_interleave(n, dim=0).unsqueeze(dim=1)

    # Helper function for processing photo groups
    def process_photos(photo_group, seq_emb_b, seq_mask_bool_b, user_side_emb_b, max_photos):
      # Photo embedding
      vid_emb = self.vid_emb(photo_group[:, :, 0])  # b*p*d
      aid_emb = self.aid_emb(photo_group[:, :, 1])  # b*p*d
      cate_two_emb = self.cate_two_emb(photo_group[:, :, 2])  # b*p*d
      cate_one_emb = self.cate_one_emb(photo_group[:, :, 3])  # b*p*d
      up_emb = self.up_type_emb(photo_group[:, :, 4])  # b*p*d

      up_wda_emb = self.wday_emb(photo_group[:, :, 5])  # b*p*d
      up_hou_emb = self.hour_emb(photo_group[:, :, 6])  # b*p*d
      up_min_emb = self.min_emb(photo_group[:, :, 7])  # b*p*d

      photo_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb], dim=2)  # b*p*5d

      # Repeat and prepare DIN inputs
      photo_3dim_repeat = photo_input.reshape(-1, 5 * self.emb_dim).unsqueeze(dim=1).expand(
        [-1, self.seq_len, -1])  # bp*seq_len*5d
      din_inputs = torch.cat([seq_emb_b, photo_3dim_repeat], dim=2)  # bp*seq_len*10d

      # DIN attention
      din_logits = self.din_mlp(din_inputs)  # bp*seq_len*1
      din_logits = torch.transpose(din_logits, 2, 1)  # bp*1*seq_len

      padding_num = -2 ** 30 + 1
      din_logits = torch.where(seq_mask_bool_b, din_logits,
                               torch.full_like(din_logits, fill_value=padding_num))  # bp*1*seq_len
      din_scores = F.softmax(din_logits, dim=2)  # bp*1*seq_len
      din_interest = torch.bmm(din_scores, seq_emb_b).squeeze()  # bp*5d

      # MLP input
      mlp_input = torch.cat([
        user_side_emb_b,
        din_interest.reshape(-1, max_photos, 5 * self.emb_dim),
        photo_input,
        up_wda_emb, up_hou_emb, up_min_emb
      ], dim=2)  # b*21d

      logits = self.mlp(mlp_input).reshape(-1, max_photos)  # b*p
      return logits

    # Process rank_pos_photos (10 photos)
    rank_pos_logits = process_photos(rank_pos_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10, max_photos=10)

    # Process rank_neg_photos (10 photos)
    rank_neg_logits = process_photos(rank_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10, max_photos=10)

    # Process coarse_neg_photos (10 photos)
    coarse_neg_logits = process_photos(coarse_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10,
                                       max_photos=10)

    # Process prerank_neg_photos (10 photos)
    prerank_neg_logits = process_photos(prerank_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10,
                                        max_photos=10)
    # print("rank_pos_logits=%s, rank_neg_logits=%s, coarse_neg_logits=%s, prerank_neg_logits=%s"%(
    #       rank_pos_logits.device, rank_neg_logits.device, coarse_neg_logits.device, prerank_neg_logits.device))

    all_lib_neg_logits = process_photos(all_lib_neg_photos, seq_emb_bn, seq_mask_bool_bn, user_side_emb_bn, max_photos=n)
    return rank_pos_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits, all_lib_neg_logits

  def forward_all_by_rerank(self, inputs):
    request_wday, request_hour, request_min, \
      uid, did, gender, age, province, \
      seq_arr, seq_mask, seq_len, \
      rerank_pos_photos, rerank_neg_photos, rank_neg_photos, coarse_neg_photos, prerank_neg_photos = inputs

    # Context embedding
    req_wda_emb = self.wday_emb(request_wday)  # b*d
    req_hou_emb = self.hour_emb(request_hour)  # b*d
    req_min_emb = self.min_emb(request_min)  # b*d

    # User embedding
    uid_emb = self.uid_emb(uid)  # b*d
    did_emb = self.did_emb(did)  # b*d
    gen_emb = self.gender_emb(gender)  # b*d
    age_emb = self.age_emb(age)  # b*d
    pro_emb = self.province_emb(province)  # b*d

    # Combine user-side embeddings
    user_side_emb = torch.cat([req_wda_emb, req_hou_emb, req_min_emb, uid_emb, did_emb, gen_emb, age_emb, pro_emb],
                              dim=1)  # b*8d
    user_side_emb_b10 = user_side_emb.unsqueeze(1).repeat([1, 10, 1])  # b*10*8d

    # Sequence embedding
    seq_mask_bool = torch.ne(seq_mask, 0)  # b*seq_len

    vid_seq_emb = self.vid_emb(seq_arr[:, :, 0])  # b*seq_len*d
    aid_seq_emb = self.aid_emb(seq_arr[:, :, 1])  # b*seq_len*d
    cate_two_seq_emb = self.cate_two_emb(seq_arr[:, :, 2])  # b*seq_len*d
    cate_one_seq_emb = self.cate_one_emb(seq_arr[:, :, 3])  # b*seq_len*d
    up_seq_emb = self.up_type_emb(seq_arr[:, :, 4])  # b*seq_len*d
    # print("{def} forward_all_by_rank. self.vid_emb.weight.device=%s vid_seq_emb.device=%s" % (
    #       self.vid_emb.weight.device, vid_seq_emb.device))
    seq_emb = torch.cat([vid_seq_emb, aid_seq_emb, cate_two_seq_emb, cate_one_seq_emb, up_seq_emb],
                        dim=2)  # b*seq_len*5d
    seq_emb_b10 = seq_emb.repeat_interleave(10, dim=0)  # b10*seq_len*5d

    seq_mask_bool_b10 = seq_mask_bool.repeat_interleave(10, dim=0).unsqueeze(dim=1)  # b10*1*seq_len
    # print("{def} forward_all_by_rank. seq_emb.device=%s"%(seq_emb.device))

    # Helper function for processing photo groups
    def process_photos(photo_group, seq_emb_b, seq_mask_bool_b, user_side_emb_b, max_photos):
      # Photo embedding
      vid_emb = self.vid_emb(photo_group[:, :, 0])  # b*p*d
      aid_emb = self.aid_emb(photo_group[:, :, 1])  # b*p*d
      cate_two_emb = self.cate_two_emb(photo_group[:, :, 2])  # b*p*d
      cate_one_emb = self.cate_one_emb(photo_group[:, :, 3])  # b*p*d
      up_emb = self.up_type_emb(photo_group[:, :, 4])  # b*p*d

      up_wda_emb = self.wday_emb(photo_group[:, :, 5])  # b*p*d
      up_hou_emb = self.hour_emb(photo_group[:, :, 6])  # b*p*d
      up_min_emb = self.min_emb(photo_group[:, :, 7])  # b*p*d

      photo_input = torch.cat([vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb], dim=2)  # b*p*5d

      # Repeat and prepare DIN inputs
      photo_3dim_repeat = photo_input.reshape(-1, 5 * self.emb_dim).unsqueeze(dim=1).expand(
        [-1, self.seq_len, -1])  # bp*seq_len*5d
      din_inputs = torch.cat([seq_emb_b, photo_3dim_repeat], dim=2)  # bp*seq_len*10d

      # DIN attention
      din_logits = self.din_mlp(din_inputs)  # bp*seq_len*1
      din_logits = torch.transpose(din_logits, 2, 1)  # bp*1*seq_len

      padding_num = -2 ** 30 + 1
      din_logits = torch.where(seq_mask_bool_b, din_logits,
                               torch.full_like(din_logits, fill_value=padding_num))  # bp*1*seq_len
      din_scores = F.softmax(din_logits, dim=2)  # bp*1*seq_len
      din_interest = torch.bmm(din_scores, seq_emb_b).squeeze()  # bp*5d

      # MLP input
      mlp_input = torch.cat([
        user_side_emb_b,
        din_interest.reshape(-1, max_photos, 5 * self.emb_dim),
        photo_input,
        up_wda_emb, up_hou_emb, up_min_emb
      ], dim=2)  # b*21d

      logits = self.mlp(mlp_input).reshape(-1, max_photos)  # b*p
      return logits

    # Process rerank_pos_photos (10 photos)
    rerank_pos_logits = process_photos(rerank_pos_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10, max_photos=10)

    # Process rerank_neg_photos (10 photos)
    rerank_neg_logits = process_photos(rerank_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10, max_photos=10)

    # Process rank_neg_photos (10 photos)
    rank_neg_logits = process_photos(rank_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10, max_photos=10)

    # Process coarse_neg_photos (10 photos)
    coarse_neg_logits = process_photos(coarse_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10,
                                       max_photos=10)

    # Process prerank_neg_photos (10 photos)
    prerank_neg_logits = process_photos(prerank_neg_photos, seq_emb_b10, seq_mask_bool_b10, user_side_emb_b10,
                                        max_photos=10)
    # print("rank_pos_logits=%s, rank_neg_logits=%s, coarse_neg_logits=%s, prerank_neg_logits=%s"%(
    #       rank_pos_logits.device, rank_neg_logits.device, coarse_neg_logits.device, prerank_neg_logits.device))
    return rerank_pos_logits, rerank_neg_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits

class DSSM(nn.Module):
  def __init__(self, emb_dim, seq_len, device, id_cnt_dict, nn_units=128):
    super(DSSM, self).__init__()

    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.nn_units = nn_units

    # user
    self.uid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['user_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['device_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['gender'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['age'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['province'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # item
    # fix1:
    video_id = id_cnt_dict.get('photo_id')
    if video_id is None:
      video_id = id_cnt_dict.get('video_id')
    self.vid_emb = nn.Embedding(
      num_embeddings=video_id + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['author_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    # fix2:
    category_level_two = id_cnt_dict.get('category_level_two')
    if category_level_two is None:
      category_level_two = id_cnt_dict.get(' category_level_two')
    self.cate_two_emb = nn.Embedding(
      num_embeddings=category_level_two + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_one'] + 2,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    self.up_type_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['upload_type'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # context
    self.wday_emb = nn.Embedding(
      num_embeddings=7 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.hour_emb = nn.Embedding(
      num_embeddings=24 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.min_emb = nn.Embedding(
      num_embeddings=60 + 2,
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
      seq_arr, seq_mask, seq_len, \
      rank_pos_photos, rank_neg_photos, coarse_neg_photos, prerank_neg_photos, all_lib_neg_photos, n = inputs

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
    rank_neg_logits = compute_photo_logits(rank_neg_photos)  # b*10
    coarse_neg_logits = compute_photo_logits(coarse_neg_photos)  # b*10
    prerank_neg_logits = compute_photo_logits(prerank_neg_photos)  # b*10

    all_lib_neg_logits = compute_photo_logits(all_lib_neg_photos)

    # Return all logits
    return rank_pos_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits, all_lib_neg_logits


  def forward_all_by_rerank(self, inputs):
    request_wday, request_hour, request_min, \
      uid, did, gender, age, province, \
      seq_arr, seq_mask, seq_len, \
      rerank_pos_photos, rerank_neg_photos, rank_neg_photos, coarse_neg_photos, prerank_neg_photos = inputs

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
    rerank_pos_logits = compute_photo_logits(rerank_pos_photos)  # b*10
    rerank_neg_logits = compute_photo_logits(rerank_neg_photos)  # b*10
    rank_neg_logits = compute_photo_logits(rank_neg_photos)  # b*10
    coarse_neg_logits = compute_photo_logits(coarse_neg_photos)  # b*10
    prerank_neg_logits = compute_photo_logits(prerank_neg_photos)  # b*10

    # Return all logits
    return rerank_pos_logits, rerank_neg_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits

class MLP(nn.Module):
  def __init__(self, emb_dim, seq_len, device, id_cnt_dict,    nn_units=128):
    super(MLP, self).__init__()

    self.emb_dim = emb_dim
    self.seq_len = seq_len
    self.device = device
    self.nn_units = nn_units

    # user
    self.uid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['user_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.did_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['device_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.gender_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['gender'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.age_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['age'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.province_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['province'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # item
    # fix1:
    video_id = id_cnt_dict.get('photo_id')
    if video_id is None:
      video_id = id_cnt_dict.get('video_id')
    self.vid_emb = nn.Embedding(
      num_embeddings=video_id + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.aid_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['author_id'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    # fix2:
    category_level_two = id_cnt_dict.get('category_level_two')
    if category_level_two is None:
      category_level_two = id_cnt_dict.get(' category_level_two')
    self.cate_two_emb = nn.Embedding(
      num_embeddings=category_level_two + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )
    self.cate_one_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['category_level_one'] + 2,
      embedding_dim=emb_dim,
      padding_idx=2
    )
    self.up_type_emb = nn.Embedding(
      num_embeddings=id_cnt_dict['upload_type'] + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # context
    self.wday_emb = nn.Embedding(
      num_embeddings=7 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.hour_emb = nn.Embedding(
      num_embeddings=24 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    self.min_emb = nn.Embedding(
      num_embeddings=60 + 2,
      embedding_dim=emb_dim,
      padding_idx=0
    )

    # encoder
    self.mlpx = nn.Sequential(
      nn.Linear(emb_dim * 21, self.nn_units),
      nn.ReLU(),
      nn.Linear(self.nn_units, self.nn_units//2),
      nn.ReLU(),
      nn.Linear(self.nn_units//2, self.nn_units//4),
      nn.ReLU(),
      nn.Linear(self.nn_units//4, 1)
    )


  def forward_all_by_rerank(self, inputs):
    request_wday, request_hour, request_min, \
      uid, did, gender, age, province, \
      seq_arr, seq_mask, seq_len, \
      rerank_pos_photos, rerank_neg_photos, rank_neg_photos, coarse_neg_photos, prerank_neg_photos = inputs

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
    
    def compute_photo_logits(u_input, photo_inputs):
      # Extract photo embeddings
      num_items = photo_inputs.size(1)
      u_expanded = u_input.unsqueeze(1).repeat(1, num_items, 1)
      
      vid_emb = self.vid_emb(photo_inputs[:, :, 0])  # b*p*d
      aid_emb = self.aid_emb(photo_inputs[:, :, 1])
      cate_two_emb = self.cate_two_emb(photo_inputs[:, :, 2])
      cate_one_emb = self.cate_one_emb(photo_inputs[:, :, 3])
      up_emb = self.up_type_emb(photo_inputs[:, :, 4])
      up_wda_emb = self.wday_emb(photo_inputs[:, :, 5])
      up_hou_emb = self.hour_emb(photo_inputs[:, :, 6])
      up_min_emb = self.min_emb(photo_inputs[:, :, 7])
      
      combined = torch.cat([u_expanded, vid_emb, aid_emb, cate_two_emb, cate_one_emb, up_emb,
                           up_wda_emb, up_hou_emb, up_min_emb], dim=2) # b,p,32
      
      b, p, l = combined.shape
      flattened = combined.view(b * p, l)
      logits = self.mlpx(flattened)
      logits = logits.view(b, p, 1).squeeze(-1)
      
      return logits

    # Compute logits for each type of photo inputs
    rerank_pos_logits = compute_photo_logits(u_input,rerank_pos_photos)  # b*10
    rerank_neg_logits = compute_photo_logits(u_input,rerank_neg_photos)  # b*10
    rank_neg_logits = compute_photo_logits(u_input,rank_neg_photos)  # b*10
    coarse_neg_logits = compute_photo_logits(u_input,coarse_neg_photos)  # b*10
    prerank_neg_logits = compute_photo_logits(u_input,prerank_neg_photos)  # b*10

    # Return all logits
    return rerank_pos_logits, rerank_neg_logits, rank_neg_logits, coarse_neg_logits, prerank_neg_logits
