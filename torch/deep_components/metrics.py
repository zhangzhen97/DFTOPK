import numpy as np

try:
    import torch
    from scipy import stats
    from scipy.stats import kendalltau
except ImportError as e:
    print(f"Import error: {e}")
else:
    print("Modules imported successfully.")



def select_samples_all_batches(
    rank_pos: torch.Tensor,
    rank_neg: torch.Tensor,
    coarse_neg: torch.Tensor,
    prerank_neg: torch.Tensor,
    n: int
) -> torch.Tensor:
    """
    从四个形状为 (bs, p, k) 的张量中，为每个 batch 随机选取 n 个不在同一 batch 的样本，
    并确保每个样本的 p 索引不重复。

    参数:
        rank_pos, rank_neg, coarse_neg, prerank_neg (Tensor): 输入张量，形状为 (bs, p, k)
        n (int): 每个 batch 要选取的样本数

    返回:
        Tensor: 形状为 (bs, n, k)
    """
    
    bs, p, k = rank_pos.shape

    # 形状为 (bs, 4*p, k)
    stacked = torch.cat([rank_pos, rank_neg, coarse_neg, prerank_neg], dim=1) 

    # 构造 mask，用于排除当前 batch 自己
    mask = torch.arange(bs, device=rank_pos.device).view(-1, 1) != torch.arange(bs, device=rank_pos.device)
    valid_batch_indices = torch.arange(bs, device=rank_pos.device).repeat(bs, 1)[mask].view(bs, bs - 1)

    # 为每个 batch 从 valid_batch_indices 中随机选择 n 个 batch
    indices_for_batch = torch.multinomial(
        torch.ones(bs, bs - 1, device=rank_pos.device), 
        n, 
        replacement=False
    )  # (bs, n)
    selected_batch_indices = valid_batch_indices.gather(1, indices_for_batch)  # (bs, n)
    print(selected_batch_indices)

    # 为每个 batch 生成 n 个不重复的 p 索引
    p_indices = torch.multinomial(
        torch.ones(bs, 4*p, device=rank_pos.device), 
        n, 
        replacement=False
    ).long()  # (bs, n)
    print(p_indices)

    # 使用高级索引选取样本
    selected = stacked[
        selected_batch_indices,  # batch 索引 (bs, n)
        p_indices,               # p 索引 (bs, n)
        :                        # 取全部 k 维
    ]  # 形状: (bs, n, k)

    return selected

def select_samples_all_batches_v1(
    rank_pos: torch.Tensor,
    rank_neg: torch.Tensor,
    coarse_neg: torch.Tensor,
    prerank_neg: torch.Tensor,
    n: int
) -> torch.Tensor:
    """
    最终修正版，严格保证：
    1. 每个batch内 (selected_batch, p_index) 对不重复
    2. 自动处理大n值情况
    3. 完全向量化实现
    """
    bs, p, k = rank_pos.shape
    device = rank_pos.device
    total_per_batch = 4 * p  # 每个batch的总候选数

    # ===== 1. 准备候选batch索引 =====
    batch_indices = torch.arange(bs, device=device).unsqueeze(0).expand(bs, bs)
    batch_indices = batch_indices[~torch.eye(bs, dtype=torch.bool, device=device)].view(bs, bs-1)
    # step 2: expand 为 (bs, bs-1, 4*p)，复制 batch 索引
    batch_indices_exp = batch_indices.unsqueeze(-1).expand(-1, -1, total_per_batch)           # (bs, bs-1, 4p)
    item_indices_exp = torch.arange(total_per_batch, device=device).view(1, 1, -1).expand(bs, bs - 1, -1)  # (bs, bs-1, 4p)

    # 4. 拉平成 (bs, (bs-1)*4p)
    flat_batch_indices = batch_indices_exp.reshape(bs, -1)  # (bs, (bs-1)*4p)
    flat_item_indices = item_indices_exp.reshape(bs, -1)    # (bs, (bs-1)*4p)

    # 5. 从 (bs-1)*4p 个候选中采样 n 个不重复的索引
    total_candidates = (bs - 1) * total_per_batch
    sample_positions = torch.multinomial(
        torch.ones(bs, total_candidates, device=device), n, replacement=False
    )  # (bs, n)

    # 6. 获取采样对应的 batch 和 item 索引
    selected_batch_ids = flat_batch_indices.gather(1, sample_positions)  # (bs, n)
    selected_item_ids = flat_item_indices.gather(1, sample_positions)    # (bs, n)

    # 7. 执行 gather，获取最终采样数据
    sampled_b = selected_batch_ids.reshape(-1)  # (bs * n,)
    sampled_i = selected_item_ids.reshape(-1)   # (bs * n,)

    # 从 stacked 中获取这些样本
    stacked = torch.cat([rank_pos, rank_neg, coarse_neg, prerank_neg], dim=1)
    sampled = stacked[sampled_b, sampled_i]     # (bs * n, dim)
    sampled = sampled.view(bs, n, k)          # (bs, n, dim)

    return sampled

def get_dcg_at_k(scores, k):
    """Compute DCG at rank K."""
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0

def get_ndcg_at_k(scores, k):
    """Compute NDCG at rank K."""
    ideal_scores = sorted(scores, reverse=True)
    dcg_max = get_dcg_at_k(ideal_scores, k)
    if not dcg_max:
        return 0.0
    return get_dcg_at_k(scores, k) / dcg_max

def get_kendall_tau_topN(logits, ranks, stages, topN = 0):
    """
    Compute Kendall's Tau for the top-N samples.

    Parameters:
        logits: np.ndarray
            Predicted logits for each batch (shape: [batch_size, N]).
        ranks: np.ndarray
            Ground truth ranking indices for each batch (shape: [batch_size, N]).
        stages: np.ndarray
            Stage labels (1 for positive, 2 for negative, others ignored) (shape: [batch_size, N]).
        topN: int
            Number of top samples to consider for Kendall's Tau computation.

    Returns:
        kendall_scores
            List of Kendall's Tau scores for each batch.
    """
    tau = None
    valid_indices = np.where((stages == 1) | (stages == 2))[0]
    if len(valid_indices) < 2:
       return tau

    valid_logits = logits[valid_indices]
    valid_ranks = ranks[valid_indices]

    sorted_indices = np.argsort(-valid_logits)
    sorted_logits = valid_logits[sorted_indices]
    sorted_ranks = valid_ranks[sorted_indices]

    if topN > 1:
        top_logits = sorted_logits[:topN]
        top_ranks = sorted_ranks[:topN]
    else:
        top_logits = sorted_logits
        top_ranks = sorted_ranks

    tau, _ = kendalltau(top_logits, top_ranks)

    return tau



def shuffle_last_dim(tensors_ls):
    """
    对输入的多个张量的最后一个维度进行相同的随机打乱。

    参数:
    *tensors: 任意数量的 PyTorch 张量，它们的最后一个维度必须相同。

    返回:
    打乱后的张量列表。
    """
    last_dim_size = tensors_ls[0].shape[-1]
    indices = np.random.permutation(last_dim_size)
    shuffled_tensors = [tensor[:, indices] for tensor in tensors_ls]

    return shuffled_tensors

def calc_ndcg_sklearn(y_score, y_true, k=None, ignore_ties=False):
    def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
        """
        Compute DCG by averaging over possible permutations of ties.

        The gain (`y_true`) of an index falling inside a tied group (in the order
        induced by `y_score`) is replaced by the average gain within this group.
        The discounted gain for a tied group is then the average `y_true` within
        this group times the sum of discounts of the corresponding ranks.

        This amounts to averaging scores for all possible orderings of the tied
        groups.

        (note in the case of dcg@k the discount is 0 after index k)

        Parameters
        ----------
        y_true : ndarray
            The true relevance scores.

        y_score : ndarray
            Predicted scores.

        discount_cumsum : ndarray
            Precomputed cumulative sum of the discounts.

        Returns
        -------
        discounted_cumulative_gain : float
            The discounted cumulative gain.

        References
        ----------
        McSherry, F., & Najork, M. (2008, March). Computing information retrieval
        performance measures efficiently in the presence of tied scores. In
        European conference on information retrieval (pp. 414-421). Springer,
        Berlin, Heidelberg.
        """
        _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
        ranked = np.zeros(len(counts))
        np.add.at(ranked, inv, y_true)
        ranked /= counts
        groups = np.cumsum(counts) - 1
        discount_sums = np.empty(len(counts))
        discount_sums[0] = discount_cumsum[groups[0]]
        discount_sums[1:] = np.diff(discount_cumsum[groups])
        return (ranked * discount_sums).sum()

    def _dcg_sample_scores(y_true, y_score, k=None, log_base=2, ignore_ties=False):
        """Compute Discounted Cumulative Gain.

        Sum the true scores ranked in the order induced by the predicted scores,
        after applying a logarithmic discount.

        This ranking metric yields a high value if true labels are ranked high by
        ``y_score``.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples, n_labels)
            True targets of multilabel classification, or true scores of entities
            to be ranked.

        y_score : ndarray of shape (n_samples, n_labels)
            Target scores, can either be probability estimates, confidence values,
            or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).

        k : int, default=None
            Only consider the highest k scores in the ranking. If `None`, use all
            outputs.

        log_base : float, default=2
            Base of the logarithm used for the discount. A low value means a
            sharper discount (top results are more important).

        ignore_ties : bool, default=False
            Assume that there are no ties in y_score (which is likely to be the
            case if y_score is continuous) for efficiency gains.

        Returns
        -------
        discounted_cumulative_gain : ndarray of shape (n_samples,)
            The DCG score for each sample.

        See Also
        --------
        ndcg_score : The Discounted Cumulative Gain divided by the Ideal Discounted
            Cumulative Gain (the DCG obtained for a perfect ranking), in order to
            have a score between 0 and 1.
        """
        discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
        if k is not None:
            discount[k:] = 0
        if ignore_ties:
            ranking = np.argsort(y_score)[:, ::-1]
            ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
            cumulative_gains = discount.dot(ranked.T)
        else:
            discount_cumsum = np.cumsum(discount)
            cumulative_gains = [
                _tie_averaged_dcg(y_t, y_s, discount_cumsum)
                for y_t, y_s in zip(y_true, y_score)
            ]
            cumulative_gains = np.asarray(cumulative_gains)
        return cumulative_gains

    def ndcg_sample_scores(y_true, y_score, k=None, ignore_ties=False):
        """Compute Normalized Discounted Cumulative Gain.

        Sum the true scores ranked in the order induced by the predicted scores,
        after applying a logarithmic discount. Then divide by the best possible
        score (Ideal DCG, obtained for a perfect ranking) to obtain a score between
        0 and 1.

        This ranking metric yields a high value if true labels are ranked high by
        ``y_score``.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples, n_labels)
            True targets of multilabel classification, or true scores of entities
            to be ranked.

        y_score : ndarray of shape (n_samples, n_labels)
            Target scores, can either be probability estimates, confidence values,
            or non-thresholded measure of decisions (as returned by
            "decision_function" on some classifiers).

        k : int, default=None
            Only consider the highest k scores in the ranking. If None, use all
            outputs.

        ignore_ties : bool, default=False
            Assume that there are no ties in y_score (which is likely to be the
            case if y_score is continuous) for efficiency gains.

        Returns
        -------
        normalized_discounted_cumulative_gain : ndarray of shape (n_samples,)
            The NDCG score for each sample (float in [0., 1.]).

        See Also
        --------
        dcg_score : Discounted Cumulative Gain (not normalized).

        """
        gain = _dcg_sample_scores(y_true, y_score, k, ignore_ties=ignore_ties)
        # Here we use the order induced by y_true so we can ignore ties since
        # the gain associated to tied indices is the same (permuting ties doesn't
        # change the value of the re-ordered y_true)
        normalizing_gain = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
        all_irrelevant = normalizing_gain == 0
        gain[all_irrelevant] = 0
        gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
        return gain

    return np.mean(ndcg_sample_scores(y_true=y_true, y_score=y_score, k=k, ignore_ties=ignore_ties))

def calc_kdt_mask(logits, labels, mask):
    try:
        all_rst = []
        mask_bool = mask.astype(bool)
        for i in range(len(logits)):
            kdt, _ = stats.kendalltau(logits[i][mask_bool[i]], labels[i][mask_bool[i]])
            if not np.isnan(kdt):
                all_rst.append(kdt)
        if len(all_rst) > 0:
            return sum(all_rst) / len(all_rst)
        else:
            return 0.0
    except Exception as e:
        print("ERROR KDT")
        return 0.0

def calc_set_recall(logits, rank_index, topk, support_m, mask):
    """
        Args:
            logits: [batchsize,ad_num]
            labels: [batchsize,ad_num]
            topk: gt_num
            support_m: recall_num
            seq_len: length of valid sequence

        Returns: set_recall: [batchsize]

        """
    labels = rank_index * mask
    optimal_indices = np.argpartition(labels, -topk, axis=-1)[:, -topk:]
    select_indices = np.argpartition(logits, -support_m, axis=-1)[:, -support_m:]
    set_recall = []
    for pred, gt in zip(select_indices, optimal_indices):
        right_num = 0
        for y in gt:
            if y in pred:
                right_num+=1
        gt_num = len(gt)
        set_recall.append(float(right_num)/float(gt_num))
    set_recall = np.array(set_recall)
    return np.mean(set_recall), set_recall

def evaluate(model, data_loader, device, is_debug=False, desc=""):
  ndcg_ls = []
  recall_ls = []
  kdt_ls = []

  if "prerank" in desc:
      top_k = 10
      support_m = 20
  else:
      top_k = 10
      support_m = 30
  print("evaluate_params. top_k=%s support_m=%s"%( top_k, support_m))

  model.eval()
  with torch.no_grad():

    iter = 0
    for inputs in data_loader:
      iter+=1

      inputs_LongTensor = [inp.to(device) for inp in inputs[:15]]
      logits_list = model.forward_all_by_rank(inputs_LongTensor)
      rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
      mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]
      logits_list_cpu = [logits.cpu().numpy() for logits in logits_list]
      rank_index_list_cpu = [rank_index.cpu().numpy() for rank_index in rank_index_list]
      mask_list_cpu = [mask.cpu().numpy() for mask in mask_list]
      logits_concat = np.concatenate(logits_list_cpu, axis=1)
      rank_index_concat = np.concatenate(rank_index_list_cpu, axis=1)
      mask_concat = np.concatenate(mask_list_cpu, axis=1)
      print("mask_concat=%s,%s"%(mask_concat.shape, mask_concat))

      if iter < 5:
          print("PRE_SHUFFLE. logits_concat=%s, rank_index_concat=%s masks=%s" % (logits_concat[0], rank_index_concat[0], mask_concat[0]))

      [logits_concat, rank_index_concat, mask_concat] = shuffle_last_dim(
          [logits_concat, rank_index_concat,mask_concat])
      ndcg = calc_ndcg_sklearn(logits_concat, rank_index_concat, k=top_k)
      recall,set_recall = calc_set_recall(logits_concat, rank_index_concat, topk=top_k, support_m=support_m, mask=mask_concat)
      if iter < 5:
          print("AFT_SHUFFLE[%s] logits=%s rank_index=%s mask=%s, topk=%s m=%s recall=%s"%(
              desc, logits_concat[0], rank_index_concat[0], mask_concat[0], top_k, support_m, set_recall[0]))
      kdt = calc_kdt_mask(logits_concat, rank_index_concat, mask_concat)

      ndcg_ls.append(ndcg)
      recall_ls.append(recall)
      kdt_ls.append(kdt)

    print("metrics\t%s\tNDCG@%s@%s\tRECALL@%s@%s\tKDT"%(desc, top_k, support_m, top_k, support_m))
    print("metrics\t%s\t%.4f\t%s\t%.4f"%(desc, np.asarray(ndcg_ls).mean(), np.asarray(recall_ls).mean(), np.asarray(kdt_ls).mean()))

def two_stage_recall(recall_logits, prerank_logits, rank_index, mask,
                     recall_topk=30, prerank_topk=20, gt_num=10):
    labels = rank_index * mask
    recall_indices = np.argpartition(recall_logits, -recall_topk, axis=-1)[:, -recall_topk:]
    optimal_indices = np.argpartition(labels, -gt_num, axis=-1)[:, -gt_num:]

    set_recall = []
    for i in range(recall_logits.shape[0]):
        prerank_logits_in_recall = prerank_logits[i, recall_indices[i]]
        prerank_indices_in_recall = np.argpartition(prerank_logits_in_recall, -prerank_topk)[-prerank_topk:]
        final_indices = recall_indices[i][prerank_indices_in_recall]
        right_num = len(set(final_indices) & set(optimal_indices[i]))
        set_recall.append(float(right_num) / float(gt_num))

    set_recall = np.array(set_recall)
    return np.mean(set_recall),set_recall

def evaluate_join(prerank_model, retrival_model, data_loader, device, desc=""):
    prerank_model.eval()
    retrival_model.eval()

    recall_ls = []
    with torch.no_grad():

        iter = 0
        for inputs in data_loader:
            
            for name, module in retrival_model.named_modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    print("retrival_model")
                    print(f"Layer: {name}")
                    print(f"  Running Mean: {module.running_mean}")
                    print(f"  Running Variance: {module.running_var}")

            for name, module in prerank_model.named_modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    print("prerank_model")
                    print(f"Layer: {name}")
                    print(f"  Running Mean: {module.running_mean}")
                    print(f"  Running Variance: {module.running_var}")

            iter += 1
            inputs_LongTensor = [inp.to(device) for inp in inputs[:15]]
            pre_logits_list = prerank_model.forward_all_by_rank(inputs_LongTensor)
            pre_logits_list_cpu = [logits.cpu().numpy() for logits in pre_logits_list]
            retr_logits_list = retrival_model.forward_all_by_rank(inputs_LongTensor)
            retr_logits_list_cpu = [logits.cpu().numpy() for logits in retr_logits_list]

            mask_list = [tensor.to(device) for tensor in inputs[-8:-4]]
            mask_list_cpu = [mask.cpu().numpy() for mask in mask_list]

            rank_index_list = [tensor.to(device) for tensor in inputs[-4:]]
            rank_index_list_cpu = [rank_index.cpu().numpy() for rank_index in rank_index_list]

            pre_logits_concat = np.concatenate(pre_logits_list_cpu, axis=1)
            retr_logits_concat = np.concatenate(retr_logits_list_cpu, axis=1)
            rank_index_concat = np.concatenate(rank_index_list_cpu, axis=1)
            mask_concat = np.concatenate(mask_list_cpu, axis=1)
            if iter < 5:
                print("evaluate_join. retr_logits_concat=%s"%retr_logits_concat)
            [retr_logits_concat, pre_logits_concat, rank_index_concat, mask_concat] = shuffle_last_dim([
                                            retr_logits_concat, pre_logits_concat, rank_index_concat, mask_concat])
            recall,set_recall = two_stage_recall(recall_logits=retr_logits_concat, prerank_logits=pre_logits_concat,
                                      rank_index=rank_index_concat, mask=mask_concat)
            recall_ls.append(recall)
            if iter < 5:
                print("DEBUG_EVAL[%s] recall_logits=%s prerank_logits=%s rank_index=%s mask=%s, recall=%s" % (
                    "joint", retr_logits_concat[0], pre_logits_concat[0], rank_index_concat[0], mask_concat[0], set_recall[0]))

        print("metrics\t%s\t%s" % (desc, "joint_recall"))
        print("metrics\t%s\t%.4f" % (desc, np.asarray(recall_ls).mean()))

def evaluate_join_3stage(rank_model, prerank_model, retrival_model, data_loader, device, desc=""):
    rank_model.eval()
    prerank_model.eval()
    retrival_model.eval()

    recall_ls = []
    with torch.no_grad():

        iter = 0
        for inputs in data_loader:
            iter += 1
            inputs_LongTensor = [inp.to(device) for inp in inputs[:16]]
            rank_logits_list = rank_model.forward_all_by_rerank(inputs_LongTensor)
            rank_logits_list_cpu = [logits.cpu().numpy() for logits in rank_logits_list]
            pre_logits_list = prerank_model.forward_all_by_rerank(inputs_LongTensor)
            pre_logits_list_cpu = [logits.cpu().numpy() for logits in pre_logits_list]
            retr_logits_list = retrival_model.forward_all_by_rerank(inputs_LongTensor)
            retr_logits_list_cpu = [logits.cpu().numpy() for logits in retr_logits_list]
            mask_list = [tensor.to(device) for tensor in inputs[-10:-5]]
            mask_list_cpu = [mask.cpu().numpy() for mask in mask_list]
            rank_index_list = [tensor.to(device) for tensor in inputs[-5:]]
            rank_index_list_cpu = [rank_index.cpu().numpy() for rank_index in rank_index_list]

            rank_logits_concat = np.concatenate(rank_logits_list_cpu, axis=1)
            pre_logits_concat = np.concatenate(pre_logits_list_cpu, axis=1)
            retr_logits_concat = np.concatenate(retr_logits_list_cpu, axis=1)
            rank_index_concat = np.concatenate(rank_index_list_cpu, axis=1)
            mask_concat = np.concatenate(mask_list_cpu, axis=1)
            if iter < 5:
                print("evaluate_join. retr_logits_concat=%s"%retr_logits_concat)

            [retr_logits_concat, pre_logits_concat, rank_logits_concat, rank_index_concat, mask_concat] = shuffle_last_dim([
                                            retr_logits_concat, pre_logits_concat, rank_logits_concat, rank_index_concat, mask_concat])
            recall,set_recall = three_stage_recall(recall_logits=retr_logits_concat, prerank_logits=pre_logits_concat,
                                                   rank_logits=rank_logits_concat,
                                                    rank_index=rank_index_concat, mask=mask_concat)
            recall_ls.append(recall)
            if iter < 5:
                print("DEBUG_EVAL[%s] recall_logits=%s prerank_logits=%s rank_index=%s mask=%s, recall=%s" % (
                    "joint", retr_logits_concat[0], pre_logits_concat[0], rank_index_concat[0], mask_concat[0], set_recall[0]))

        print("metrics\t%s\t%s" % (desc, "joint_recall"))
        print("metrics\t%s\t%.4f" % (desc, np.asarray(recall_ls).mean()))

def three_stage_recall(recall_logits, prerank_logits, rank_logits, rank_index, mask,
                       recall_topk=40, prerank_topk=30, rank_topk=20, gt_num=10):
    labels = rank_index * mask
    batch_size = labels.shape[0]
    optimal_indices_list = []
    for i in range(batch_size):
        valid_mask = mask[i].astype(bool)
        valid_labels = labels[i][valid_mask]
        valid_original_indices = np.where(valid_mask)[0]
        sorted_indices = np.argsort(valid_labels)[::-1]
        top_indices = sorted_indices[:gt_num]
        top_original_indices = valid_original_indices[top_indices]
        optimal_indices_list.append(top_original_indices)
    optimal_indices = np.array(optimal_indices_list)

    set_recall = []
    for i in range(batch_size):
        recall_candidates = np.argpartition(recall_logits[i], -recall_topk)[-recall_topk:]
        prerank_logits_in_recall = prerank_logits[i][recall_candidates]
        prerank_candidates = recall_candidates[np.argpartition(prerank_logits_in_recall, -prerank_topk)[-prerank_topk:]]
        rank_logits_in_prerank = rank_logits[i][prerank_candidates]
        final_indices = prerank_candidates[np.argpartition(rank_logits_in_prerank, -rank_topk)[-rank_topk:]]
        right_num = len(set(final_indices) & set(optimal_indices[i]))
        set_recall.append(float(right_num) / float(gt_num))
    print("set_recall=%s"%set_recall)
    set_recall = np.array(set_recall)
    return np.mean(set_recall), set_recall

def evaluate_3stage(model, data_loader, device, is_debug=False, desc=""):
  ndcg_ls = []
  recall_ls = []
  kdt_ls = []
  if "\trank" in desc:
      top_k = 10
      support_m = 20
  elif "\tprerank" in desc:
      top_k = 10
      support_m = 30
  else:
      # top_k = 20
      top_k = 10
      support_m = 40
  print("evaluate_params. top_k=%s support_m=%s"%( top_k, support_m))

  model.eval()
  with torch.no_grad():

    iter = 0
    for inputs in data_loader:
      iter+=1
      inputs_LongTensor = [inp.to(device) for inp in inputs[:16]]
      logits_list = model.forward_all_by_rerank(inputs_LongTensor)
      mask_list = [tensor.to(device) for tensor in inputs[-10:-5]]
      rank_index_list = [tensor.to(device) for tensor in inputs[-5:]]
      logits_list_cpu = [logits.cpu().numpy() for logits in logits_list]
      rank_index_list_cpu = [rank_index.cpu().numpy() for rank_index in rank_index_list]
      mask_list_cpu = [mask.cpu().numpy() for mask in mask_list]
      logits_concat = np.concatenate(logits_list_cpu, axis=1)
      rank_index_concat = np.concatenate(rank_index_list_cpu, axis=1)
      mask_concat = np.concatenate(mask_list_cpu, axis=1)
      print("mask_concat=%s,%s"%(mask_concat.shape, mask_concat))

      if iter < 5:
          print("PRE_SHUFFLE. logits_concat=%s, rank_index_concat=%s masks=%s" % (logits_concat[0], rank_index_concat[0], mask_concat[0]))

      [logits_concat, rank_index_concat, mask_concat] = shuffle_last_dim(
          [logits_concat, rank_index_concat, mask_concat])
      ndcg = calc_ndcg_sklearn(logits_concat, rank_index_concat, k=top_k)
      recall,set_recall = calc_set_recall(logits_concat, rank_index_concat, topk=top_k, support_m=support_m, mask=mask_concat)
      if iter < 5:
          print("AFT_SHUFFLE[%s] logits=%s rank_index=%s mask=%s, topk=%s m=%s recall=%s"%(
              desc, logits_concat[0], rank_index_concat[0], mask_concat[0], top_k, support_m, set_recall[0]))
      kdt = calc_kdt_mask(logits_concat, rank_index_concat, mask_concat)

      ndcg_ls.append(ndcg)
      recall_ls.append(recall)
      kdt_ls.append(kdt)

    print("metrics\t%s\tNDCG@%s@%s\tRECALL@%s@%s\tKDT"%(desc, top_k, support_m, top_k, support_m))
    print("metrics\t%s\t%.4f\t%s\t%.4f"%(desc, np.asarray(ndcg_ls).mean(), np.asarray(recall_ls).mean(), np.asarray(kdt_ls).mean()))

def check_three_stage_recall():
    batch_size = 1
    seq_length = 50
    gt_num = 10
    recall_logits = np.zeros((batch_size, seq_length))
    for i in range(batch_size):
        recall_logits[i, :40] = np.arange(40, 0, -1)
    recall_logits[0][0],recall_logits[0][-1] = recall_logits[0][-1],recall_logits[0][0]

    prerank_logits = np.zeros((batch_size, seq_length))
    for i in range(batch_size):
        prerank_logits[i, :30] = np.arange(30, 0, -1)
    prerank_logits[0][8],prerank_logits[0][-1] = prerank_logits[0][-1],prerank_logits[0][8]

    rank_logits = np.zeros((batch_size, seq_length))
    for i in range(batch_size):
        rank_logits[i, :20] = np.arange(20, 0, -1)

    rank_index = np.zeros((batch_size, seq_length))
    for i in range(batch_size):
        rank_index[i, :gt_num] = 1

    mask = np.zeros((batch_size, seq_length))
    for i in range(batch_size):
        mask[i, :40] = 1 

    mean_recall, _ = three_stage_recall(
        recall_logits=recall_logits,
        prerank_logits=prerank_logits,
        rank_logits=rank_logits,
        rank_index=rank_index,
        mask=mask,
        recall_topk=40,
        prerank_topk=30,
        rank_topk=20,
        gt_num=10
    )

    print(f"Expected Recall: 1.0000, Actual Recall: {mean_recall:.4f}")
    assert np.isclose(mean_recall, 0.8), "Recall should be 1.0 for this test case"

def evaluate_join_negativesampling(prerank_model, retrival_model, data_loader, device, samlpe_num, desc=""):
    prerank_model.eval()
    retrival_model.eval()

    recall_ls = []
    with torch.no_grad():

        iter = 0
        for inputs in data_loader:

            iter += 1
            # batch 内负采样
            rank_pos, rank_neg, coarse_neg, prerank_neg = inputs[11:15]
            rank_index_list = inputs[-4:]
            mask_list = inputs[-8:-4]
            n = samlpe_num  # sample nums
            if n >0:
                all_lib_rank = torch.zeros(rank_index_list[0].shape[0], n)
                all_lib_mask = torch.ones(inputs[-8].shape[0],n)
                all_lib_neg_photo = select_samples_all_batches_v1(rank_pos, rank_neg, coarse_neg, prerank_neg, n)
            feature_input_list = inputs[:15]
            if n >0:
                feature_input_list.append(all_lib_neg_photo)
                rank_index_list.append(all_lib_rank)
                mask_list.append(all_lib_mask)
            # print('mask_size',mask_list[-1].shape[1])
            inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in feature_input_list]
            if n >0:
                inputs_LongTensor.append(n)
            # inputs_LongTensor = [inp.to(device) for inp in inputs[:15]]
            pre_logits_list = prerank_model.forward_all_by_rank(inputs_LongTensor)
            pre_logits_list_cpu = [logits.cpu().numpy() for logits in pre_logits_list]
            retr_logits_list = retrival_model.forward_all_by_rank(inputs_LongTensor)
            retr_logits_list_cpu = [logits.cpu().numpy() for logits in retr_logits_list]

            mask_list_cpu = [mask.cpu().numpy() for mask in mask_list]
            rank_index_list_cpu = [rank_index.cpu().numpy() for rank_index in rank_index_list]

            pre_logits_concat = np.concatenate(pre_logits_list_cpu, axis=1)
            retr_logits_concat = np.concatenate(retr_logits_list_cpu, axis=1)
            rank_index_concat = np.concatenate(rank_index_list_cpu, axis=1)
            mask_concat = np.concatenate(mask_list_cpu, axis=1)
            if iter < 5:
                print("evaluate_join. retr_logits_concat=%s"%retr_logits_concat)
            [retr_logits_concat, pre_logits_concat, rank_index_concat, mask_concat] = shuffle_last_dim([
                                            retr_logits_concat, pre_logits_concat, rank_index_concat, mask_concat])
            recall,set_recall = two_stage_recall(recall_logits=retr_logits_concat, prerank_logits=pre_logits_concat,
                                      rank_index=rank_index_concat, mask=mask_concat)
            recall_ls.append(recall)
            if iter < 5:
                print("DEBUG_EVAL[%s] recall_logits=%s prerank_logits=%s rank_index=%s mask=%s, recall=%s" % (
                    "joint", retr_logits_concat[0], pre_logits_concat[0], rank_index_concat[0], mask_concat[0], set_recall[0]))

        print("metrics\t%s\t%s" % (desc, "joint_recall"))
        print("metrics\t%s\t%.4f" % (desc, np.asarray(recall_ls).mean()))

def evaluate_negativesampling(model, data_loader, device,sample_num, is_debug=False,desc=""):
    ndcg_ls = []
    recall_ls = []
    kdt_ls = []
    if "prerank" in desc:
        top_k = 10
        support_m = 20
    else:
        top_k = 10
        support_m = 30
    print("evaluate_params. top_k=%s support_m=%s"%( top_k, support_m))
    model.eval()
    with torch.no_grad():

        iter = 0
        for inputs in data_loader:
            iter+=1
            # batch 内负采样
            rank_pos, rank_neg, coarse_neg, prerank_neg = inputs[11:15]
            rank_index_list = inputs[-4:]
            mask_list = inputs[-8:-4]
            n = sample_num # sample nums
            if n >0:
                all_lib_rank = torch.zeros(rank_index_list[0].shape[0], n)
                all_lib_mask = torch.ones(inputs[-8].shape[0],n)
                all_lib_neg_photo = select_samples_all_batches_v1(rank_pos, rank_neg, coarse_neg, prerank_neg, n)
            feature_input_list = inputs[:15]
            if n >0:
                feature_input_list.append(all_lib_neg_photo)
                rank_index_list.append(all_lib_rank)
                mask_list.append(all_lib_mask)
            # print('mask_size',mask_list[-1].shape[1])
            inputs_LongTensor = [torch.LongTensor(inp.numpy()).to(device) for inp in feature_input_list]
            if n > 0:
                inputs_LongTensor.append(n)
            #
            logits_list = model.forward_all_by_rank(inputs_LongTensor)
            rank_index_list = [tensor.to(device) for tensor in rank_index_list]
            mask_list = [tensor.to(device) for tensor in mask_list]

            logits_list_cpu = [logits.cpu().numpy() for logits in logits_list]
            rank_index_list_cpu = [rank_index.cpu().numpy() for rank_index in rank_index_list]
            mask_list_cpu = [mask.cpu().numpy() for mask in mask_list]

            logits_concat = np.concatenate(logits_list_cpu, axis=1)
            rank_index_concat = np.concatenate(rank_index_list_cpu, axis=1)
            mask_concat = np.concatenate(mask_list_cpu, axis=1)
            print("mask_concat=%s,%s"%(mask_concat.shape, mask_concat))


            if iter < 5:
                print("PRE_SHUFFLE. logits_concat=%s, rank_index_concat=%s masks=%s" % (logits_concat[0], rank_index_concat[0], mask_concat[0]))

            [logits_concat, rank_index_concat, mask_concat] = shuffle_last_dim(
                [logits_concat, rank_index_concat,mask_concat])
            
            ndcg = calc_ndcg_sklearn(logits_concat, rank_index_concat, k=top_k)
            recall,set_recall = calc_set_recall(logits_concat, rank_index_concat, topk=top_k, support_m=support_m, mask=mask_concat)
            if iter < 5:
                print("AFT_SHUFFLE[%s] logits=%s rank_index=%s mask=%s, topk=%s m=%s recall=%s"%(
                    desc, logits_concat[0], rank_index_concat[0], mask_concat[0], top_k, support_m, set_recall[0]))
            kdt = calc_kdt_mask(logits_concat, rank_index_concat, mask_concat)

            ndcg_ls.append(ndcg)
            recall_ls.append(recall)
            kdt_ls.append(kdt)

        print("metrics\t%s\tNDCG@%s@%s\tRECALL@%s@%s\tKDT"%(desc, top_k, support_m, top_k, support_m))
        print("metrics\t%s\t%.4f\t%s\t%.4f"%(desc, np.asarray(ndcg_ls).mean(), np.asarray(recall_ls).mean(), np.asarray(kdt_ls).mean()))

if __name__ == "__main__":
    check_three_stage_recall()