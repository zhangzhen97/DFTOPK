import os
if os.environ.get('tf_v1')=="T":
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
else:
    import tensorflow as tf
import sys
import traceback
sys.path.append("/utils")

def tf_get_tensor_rank(input): # 原tf_shape_len, 获取tensor的秩
    if input is None:
        return 0
    return len(input.get_shape().as_list())

def pad_sequence(seq, maxlen, dtype, padding='post', truncating='post', value=0.0):
    padding_len = tf.math.maximum(maxlen-tf.shape(seq)[1],0)
    if padding=='post':
        pad_seq = tf.pad(seq,[[0,0],[0,padding_len]],constant_values=tf.constant(value,dtype=dtype))
    else:
        pad_seq = tf.pad(seq, [[0, 0], [padding_len,0]], constant_values=tf.constant(value, dtype=dtype))
    if truncating=='post':
        pad_seq = pad_seq[:,:maxlen]
    else:
        pad_seq = pad_seq[:,-maxlen:]
    return pad_seq


def batch_1d_data_preprocess_for_ltr(logits, pv, labels, logger, padding_to_len=None, extend_tensors_to_group_by=None,
                                     sample_mask=False):
    '''
            输入数据为1d的<user,pvid,item>数据，将其group by之后输出以pvid聚合的数据，shape为[batchsize,max_len]
            会给非填充的label+0.0001, 保证label非0
            :param logits: 样本logits
            :param labels: 样本label 比如 精排cpm
            :param pv:     样本pv_id(llsid)
            :return: logits_idx, labels_idx, count: [batch_size]即seq_len
    '''
    try:
        logger.info("{def} def_pair_loss. logits=%s labels=%s pv=%s" % (logits, labels, pv))

        # labels = tf.constant([[12],[11], [3], [8], [1.5], [4.3], [129], [40], [37]])  # cpm
        # pv = tf.constant([111, 111, 111, 111, 222, 222, 333, 333, 333])
        # logits = tf.constant([[9.2, 9.0, 8.2, 9.8, 0.3, 0.6, 0.47, 0.77, 0.97]])

        # 1.get unique statics for batch: llsid, idx_ls, llsid_count
        batch_size = tf.shape(pv)[0]
        if tf_get_tensor_rank(logits) > 1:
            logits = tf.reshape(logits, [batch_size])
        if tf_get_tensor_rank(labels) > 1:
            labels = tf.reshape(labels, [batch_size])
        if extend_tensors_to_group_by is not None:
            for key in extend_tensors_to_group_by.keys():
                # if key in ['recall_sample_flag', 'rank_flag', 'pre_rank_flag', 'stage_info', 'queue_info',
                #            'strategy_info', "soft_rank_num", "soft_prerank_num", "soft_recall_num"]:
                #     extend_tensors_to_group_by[key] = tf.reshape(extend_tensors_to_group_by[key], [batch_size])
                # if tf_get_tensor_rank(extend_tensors_to_group_by[key]):
                extend_tensors_to_group_by[key] = tf.reshape(extend_tensors_to_group_by[key], [batch_size])
        # # 让label非零约束.
        # labels = labels + 0.0001
        logger.info("{def_pair_loss} DEBUG. labels=%s" % labels)
        logger.info("{def_pair_loss} DEBUG. logits=%s" % logits)

        rank_key, rank_idx, count = tf.unique_with_counts(pv)
        max_count = tf.reduce_max(count)
        pv_num = tf.shape(count)[0]
        logger.info("{def_pair_loss} DEBUG. rank_key=%s" % rank_key)
        logger.info("{def_pair_loss} DEBUG. rank_idx=%s" % rank_idx)
        logger.info("{def_pair_loss} DEBUG. count=%s" % count)

        # 2. collect exmales for each pv[pv_num * max_item_num]
        output_shape_emb = [pv_num, max_count]
        # <pv_num>*example_idx [pv_num, batch_size]
        rank_idx_tile = tf.tile(tf.expand_dims(rank_idx, 0), [pv_num, 1])
        # pv_idx*<batch_size>  [pv_num, batch_size]
        range_count_tile = tf.tile(tf.expand_dims(tf.range(pv_num), 1), [1, batch_size])
        # pv example mask[pv_num, batch_size]
        pv_example_mask = tf.cast(tf.equal(rank_idx_tile, range_count_tile), tf.int32)
        # pv pre example nums[pv_num, batch_size]
        pv_pre_example_nums = tf.cumsum(pv_example_mask, axis=1, exclusive=True)
        masked_list_id = pv_example_mask * pv_pre_example_nums
        # get col idx:
        col_cor = tf.reduce_sum(masked_list_id, axis=0)
        # collect pv examples by row idx and col idx:
        row_cor = tf.expand_dims(rank_idx, 1)
        col_cor = tf.expand_dims(col_cor, 1)
        batch_idx = tf.concat([row_cor, col_cor], axis=1)
        # batch_idx =[[0, 0],
        #             [0, 1],
        #             [0, 2],
        #             [0, 3],
        #             [1, 0],
        #             [1, 1],
        #             [2, 0],
        #             [2, 1],
        #             [2, 2]]
        logits_idx = tf.scatter_nd(updates=logits, indices=batch_idx, shape=output_shape_emb, name="logits_idx")
        # logits_idx =[[9.2 , 9.  , 8.2 , 9.8 ],
        #              [0.3 , 0.6 , 0.  , 0.  ],
        #              [0.47, 0.77, 0.97, 0.  ]]
        labels_idx = tf.scatter_nd(updates=labels, indices=batch_idx, shape=output_shape_emb, name="labels_idx")
        # labels_idx = [[ 12.0001,  11.0001,   3.0001,   8.0001],
        #            [  1.5001,   4.3001,   0.    ,   0.    ],
        #            [129.0001,  40.0001,  37.0001,   0.    ]]
        logger.info("{def_pair_loss} DEBUG. output_shape_emb=%s" % output_shape_emb)
        logger.info("{def_pair_loss} DEBUG. logits_idx=%s" % logits_idx)
        logger.info("{def_pair_loss} DEBUG. labels_idx=%s" % labels_idx)

        rst_of_extend_tensors_to_group_by = {}
        if extend_tensors_to_group_by is not None:
            for key in extend_tensors_to_group_by.keys():
                rst_of_extend_tensors_to_group_by['%s_idx' % key] = tf.scatter_nd(
                    updates=extend_tensors_to_group_by[key],
                    indices=batch_idx,
                    shape=output_shape_emb,
                    name="%s_idx" % key)
                logger.info(
                    "{def_pair_loss} DEBUG. %s_idx = %s" % (key, rst_of_extend_tensors_to_group_by['%s_idx' % key]))

        if padding_to_len is not None:
            logits_idx = pad_sequence(logits_idx, maxlen=padding_to_len,
                                      dtype=tf.float32, padding='post', truncating='post',
                                      value=0.0)
            labels_idx = pad_sequence(labels_idx, maxlen=padding_to_len,
                                      dtype=tf.float32, padding='post', truncating='post',
                                      value=0.0)
            if extend_tensors_to_group_by is not None:
                for key in rst_of_extend_tensors_to_group_by.keys():
                    rst_of_extend_tensors_to_group_by[key] = pad_sequence(rst_of_extend_tensors_to_group_by[key],
                                                                          maxlen=padding_to_len,
                                                                          dtype=tf.float32,
                                                                          padding='post',
                                                                          truncating='post',
                                                                          value=0.0)

            count = tf.cast(tf.where(count > padding_to_len, tf.ones_like(count) * padding_to_len, count), tf.int32)
        mask = tf.sequence_mask(count, maxlen=padding_to_len)
        not_mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)
        logits_idx = logits_idx - not_mask * 1e5
        labels_idx = labels_idx * tf.cast(mask, tf.float32)

        if sample_mask:
            labels_idx_sum = tf.reduce_sum(labels_idx, axis=-1)
            count = tf.where(labels_idx_sum < 1.0, tf.zeros_like(count), count)

        if extend_tensors_to_group_by is not None:
            extend_tensors_to_group_by.update(rst_of_extend_tensors_to_group_by)
        return logits_idx, labels_idx, count

    except Exception as ex:
        print('[In batch_1d_data_preprocess_for_ltr] An error occurred: %s' % ex)
        traceback.print_exc()
        return tf.stack([logits, logits], axis=1), tf.stack([labels, labels], axis=1), tf.ones_like(pv)