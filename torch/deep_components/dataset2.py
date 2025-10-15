import os
import gc
import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import load_pkl

class Rank_Train_All_BY_RANK_Dataset(Dataset):
  def __init__(
          self,
          path_to_csv,
          seq_len,
          path_to_seq,
          path_to_request,
          is_debug = False,
          rank_offset = 0,
  ):
    t1 = time.time()
    print("dataset.INIT path_to_csv=%s"%path_to_csv)
    print("dataset.INIT rank_offset=%s"%rank_offset)
    self.date = os.path.splitext(os.path.basename(path_to_csv))[0]

    raw_df = pd.read_feather(path_to_csv)
    df = raw_df[
      ["request_id", "user_id", "request_timestamp", "device_id", "age", "gender", "province",
       ]].drop_duplicates()

    data_begin = time.time()
    self.data = df.to_numpy().copy()
    self.is_debug = is_debug

    self.seq_len = seq_len
    self.today_seq = load_pkl(path_to_seq)
    self.request_dict = load_pkl(path_to_request)
    data_used = time.time() - data_begin
    print("data_pd, time cost=%s " % data_used)

    idx_begin = time.time()
    self.rank_index_df = self.get_rank_index_pd(raw_df)
    idx_used = time.time()  - idx_begin
    print("rank_index_pd, time cost=%s"%idx_used)

    del df
    del raw_df
    gc.collect()

    t2 = time.time()
    print(f'init data time: {t2 - t1}')

    self.search_hit_num=0
    self.search_miss_num=0

    self.rank_offset = rank_offset

  def search_rank_index_list(self, request_id, video_ids, default_v = 500):
    rank_index_ls = []
    for video_id in video_ids:
      key_to_lookup = (request_id, video_id)
      val = self.rank_index_df.get(key_to_lookup)
      if val is None:
        val = default_v
        self.search_miss_num+=1
      else:
        self.search_hit_num+=1
      rank_index_ls.append(val[0])

    rank_index_ls = np.asarray(rank_index_ls)
    return rank_index_ls

  def get_rank_index_pd(self, df):
    df = df[df["rank_pos"] == 1]
    grouped = df.groupby(["request_id", "video_id"])["rank_index"]
    print("get_rank_index_pd. df.head=%s"%df.head(100))
    print("get_rank_index_pd. grouped.head=%s"%grouped.head(100))
    rank_index_lookup = grouped.apply(list).to_dict()

    return rank_index_lookup

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    request_id = self.data[idx][0]
    request_ts = self.data[idx][2]
    request_ts_struct = time.localtime(request_ts // 1000)
    request_ts_wday = request_ts_struct.tm_wday + 1
    request_ts_hour = request_ts_struct.tm_hour + 1
    request_ts_min = request_ts_struct.tm_min + 1
    
    uid = self.data[idx][1] + 1
    did = self.data[idx][3] + 1
    age = self.data[idx][4] + 1
    gender = self.data[idx][5] + 1
    province = self.data[idx][6] + 1

    seq_full = self.today_seq[request_id][:, [0, 1, 2, 3, 4, 7]].copy()

    seq_mask = (seq_full[:, 5] > 0).astype(np.int8)
    seq_len = np.sum(seq_mask)

    seq_arr = seq_full[:, :5]
    if seq_len > 0:
      seq_arr[-seq_len:, :3] += 1
      seq_arr[-seq_len:, 4] += 1

    max_sz = 10
    rank_pos_photos, rank_pos_mask, rank_pos_rank_index = self.initialize_photo_data(max_sz, default_rank_index=0)

    if 'rank_pos' in self.request_dict[request_id] and len(self.request_dict[request_id]['rank_pos']) > 0:
      rank_pos_flow_photos = self.request_dict[request_id]['rank_pos'].copy()[:max_sz]
      self.fill_photo_data(rank_pos_photos, rank_pos_mask, rank_pos_flow_photos, max_photos=max_sz)
      rank_index_ls = self.search_rank_index_list(request_id, rank_pos_flow_photos[:, 0])
      rank_pos_rank_index[:len(rank_pos_flow_photos)] = rank_index_ls

    rank_neg_photos, rank_neg_mask, rank_neg_rank_index = self.initialize_photo_data(max_sz, default_rank_index=23+self.rank_offset)
    if 'rank_neg' in self.request_dict[request_id] and len(self.request_dict[request_id]['rank_neg']) > 0:
      rank_neg_flow_photos = self.request_dict[request_id]['rank_neg'].copy()[:max_sz]
      self.fill_photo_data(rank_neg_photos, rank_neg_mask, rank_neg_flow_photos, max_photos=max_sz)

    coarse_neg_photos, coarse_neg_mask, coarse_neg_rank_index = self.initialize_photo_data(max_sz, default_rank_index=24+self.rank_offset*2)
    if 'coarse_neg' in self.request_dict[request_id]:
      coarse_neg_flow_photos = self.request_dict[request_id]['coarse_neg'].copy()
      self.fill_photo_data(coarse_neg_photos, coarse_neg_mask, coarse_neg_flow_photos, max_photos=max_sz)
    else:
      if self.is_debug:
        print("coarse_neg[%s]= MISS"%(request_id))

    prerank_neg_photos, prerank_neg_mask, prerank_neg_rank_index = self.initialize_photo_data(max_sz, default_rank_index=25+self.rank_offset*3)
    if 'prerank_neg' in self.request_dict[request_id]:
      prerank_neg_flow_photos = self.request_dict[request_id]['prerank_neg'].copy()[:max_sz]
      self.fill_photo_data(prerank_neg_photos, prerank_neg_mask, prerank_neg_flow_photos, max_photos=max_sz)
    else:
      if self.is_debug:
        print("prerank_neg[%s]= MISS"%(request_id))
    if 30 > 25+self.rank_offset*3:
      max_rank_index = 30
    else:
      max_rank_index = 25 + self.rank_offset*3 + 1

    return request_ts_wday, request_ts_hour, request_ts_min, \
      uid, did, gender, age, province, \
      seq_arr, seq_mask, max(seq_len, 1), \
      rank_pos_photos, rank_neg_photos, coarse_neg_photos, prerank_neg_photos, \
      rank_pos_mask, rank_neg_mask, coarse_neg_mask, prerank_neg_mask, \
      max_rank_index - rank_pos_rank_index, max_rank_index - rank_neg_rank_index, \
      max_rank_index - coarse_neg_rank_index, max_rank_index - prerank_neg_rank_index

  def initialize_photo_data(self, num_photos, default_rank_index):
    photos = np.zeros(shape=[num_photos, 8], dtype=np.int64)
    photos[:, 3] = 2
    mask = np.zeros(shape=(num_photos), dtype=np.int8)
    rank_index = np.zeros(shape=(num_photos), dtype=np.int64) + default_rank_index
    return photos, mask, rank_index

  def fill_photo_data(self, photos, mask, flow_photos, max_photos):
    n_photos = min(flow_photos.shape[0], max_photos)
    if n_photos > 0:
      sampling_flow_photos = flow_photos[:n_photos]

      photos[:n_photos, 0] = sampling_flow_photos[:, 0] + 1
      photos[:n_photos, 1] = sampling_flow_photos[:, 1] + 1
      photos[:n_photos, 2] = sampling_flow_photos[:, 3] + 1
      photos[:n_photos, 3] = sampling_flow_photos[:, 6]
      photos[:n_photos, 4] = sampling_flow_photos[:, 4] + 1

      sampling_upt_lst = sampling_flow_photos[:, 5].tolist()
      photos[:n_photos, 5] = np.array(
        list(map(lambda x: time.localtime(x).tm_wday + 1, sampling_upt_lst)),
        dtype=np.int64)
      photos[:n_photos, 6] = np.array(
        list(map(lambda x: time.localtime(x).tm_hour + 1, sampling_upt_lst)),
        dtype=np.int64)
      photos[:n_photos, 7] = np.array(
        list(map(lambda x: time.localtime(x).tm_min + 1, sampling_upt_lst)),
        dtype=np.int64)

      mask[:n_photos] = 1
