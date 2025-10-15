import tensorflow as tf
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loss.LCRON_via_permutation import LCRON
from utils.data_preprocess import batch_1d_data_preprocess_for_ltr

LOG_FORMAT = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

max_sample_num = 4
joint_loss_conf = type("", (), {"prerank_model_name": 'joint/prerank_model',
                                "recall_model_name": 'joint/recall_model',
                                "joint_recall_k": 3,
                                "joint_prerank_k": 2,
                                "gt_num": 1,
                                "global_size": max_sample_num})

def get_data_group_by_pv(pred_score, labels, pv, max_sample_num, hook_prefix='',
                            extend_tensors_to_group_by=None, sample_mask=False):
    grouped_logits, grouped_labels, count = batch_1d_data_preprocess_for_ltr(
        pred_score, pv, labels, logger, padding_to_len=max_sample_num,
        extend_tensors_to_group_by=extend_tensors_to_group_by, sample_mask=sample_mask)
    return grouped_logits, grouped_labels, count

model_outputs,label_infos = dict(),dict()

labels = tf.constant([[12],[11], [3], [8], [1.5], [4.3], [129], [40], [37]])  # cpm
pv = tf.constant([111, 111, 111, 111, 222, 222, 333, 333, 333])
prerank_logits = tf.constant([[9.2, 9.0, 8.2, 9.8, 0.3, 0.6, 0.47, 0.77, 0.97]])
recall_logits = tf.constant([[9.2, 9.0, 10.2, 9.8, 0.3, 0.9, 0.27, 0.77, 0.97]])

extend_tensors_to_group_by = dict()
extend_tensors_to_group_by['recall_logits'] = recall_logits
grouped_prerank_logits, grouped_labels, count = get_data_group_by_pv(prerank_logits, labels, pv, max_sample_num, hook_prefix='',
                            extend_tensors_to_group_by=extend_tensors_to_group_by)

label_infos['label_mask'] = tf.cast(tf.sequence_mask(count, maxlen=max_sample_num), dtype=tf.float32)
label_infos['count'] = count
model_outputs["grouped_labels"] = grouped_labels
model_outputs["grouped_prerank_logits"] = grouped_prerank_logits
model_outputs["grouped_recall_logits"] = extend_tensors_to_group_by['recall_logits_idx']

loss_instance = LCRON(name='joint/cascade_model', 
                    label_infos=label_infos,
                    model_outputs=model_outputs,
                    loss_conf=joint_loss_conf,
                    logger=logger,
                    sort_op='neural_sort',
                    use_name_as_scope=True,
                    is_debug=True,
                    is_train=True)

total_loss = loss_instance.get_loss('joint/cascade_model')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())  
    loss_val = sess.run(total_loss)
    logger.info(f"[TEST] Final Loss Value: {loss_val:.6f}")