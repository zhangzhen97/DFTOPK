loss_type=$2
cuda=$3
if [ -n "$4" ]; then
    tau=$4
else
    tau="1.0"
fi

if [ -n "$5" ]; then
    epochs=$5
else
    epochs="0"
fi

if [ -n "$6" ]; then
    emb_dim=$6
else
    emb_dim=8
fi

if [ -n "$7" ]; then
    sample_num=$7
else
    sample_num=0
fi

if [ -n "$8" ]; then
    sort_type=$8
else
    sort_type="neural_sort"
fi


if [ -n "$9" ]; then
    lr=$9
else
    lr=1e-2
fi

if [ -n "${10}" ]; then
    root_path=${10}
else
    root_path="."
fi

if [ -n "${11}" ]; then
    BS=${11}
else
    BS=1024
fi


 
echo "tau=${tau}"
echo "epochs=${epochs}"
echo "lr=${lr}"
echo "BS=${BS}"
echo "epochs=${epochs}"
echo "root_path=${root_path}"
echo "sample_num=${sample_num}"

tag=${loss_type}-1st

for epoch in $epochs; do
  echo "epoch=${epoch}"
  TRAIN_PATH="${root_path}/logs_nday/TRAIN_bs-${BS}_lr-${lr}_tau${tau}_${tag}_E${epoch}_SN${sample_num}_S2.log"
  TEST_PATH="${root_path}/logs_nday/TEST_bs-${BS}_lr-${lr}_tau${tau}_${tag}_E${epoch}_SN${sample_num}_S2.log"

  if [[ "$1" == "all" || "$1" == "train" ]]; then
    echo "TRAIN_PATH=${TRAIN_PATH}"
    /share/ad/zq3/lcron/python3_7/bin/python -B -u deep_components/run_train2_negativesampling_nday.py \
    --epochs=${epoch} \
    --loss_type=${loss_type} \
    --tau="${tau}" \
    --batch_size=${BS} \
    --infer_realshow_batch_size=${BS} \
    --infer_recall_batch_size=${BS} \
    --emb_dim=${emb_dim} \
    --lr=${lr} \
    --seq_len=50 \
    --cuda=${cuda} \
    --root_path=${root_path} \
    --print_freq=100 \
    --sort_type="${sort_type}" \
    --sample_num=${sample_num} \
    --tag=${tag} > $TRAIN_PATH 2>&1
  fi

  if [[ "$1" == "all" || "$1" == "test" ]];
   then
    /share/ad/zq3/lcron/python3_7/bin/python -B -u deep_components/run_test2_negativesampling_exp_nday.py \
    --epochs=${epoch} \
    --loss_type=${loss_type} \
    --tau="${tau}" \
    --batch_size=${BS} \
    --infer_realshow_batch_size=${BS} \
    --infer_recall_batch_size=${BS} \
    --emb_dim=${emb_dim} \
    --lr=${lr} \
    --seq_len=50 \
    --cuda=${cuda} \
    --root_path=${root_path} \
    --print_freq=100 \
    --sample_num=${sample_num} \
    --tag=${tag} > $TEST_PATH 2>&1

    # collect metrics
    # sh two_stage/run_collect.sh ${loss_type}

  fi

done
