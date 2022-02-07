#!/bin/bash

# Parameters ------------------------------------------------------
TASK="multiwoz22"
DATA_DIR="data/MULTIWOZ2.2"
#CUDA_VISIBLE_DEVICES=2


for target in "taxi"; do

  OUT_DIR="sigir2_${TASK}_${target}_${opt}_v${version}_e${ss}_fs${fs}"
  mkdir -p ${OUT_DIR}

  #   Main ------------------------------------------------------------
  for step in train test; do
      args_add=""
      if [ "$step" = "train" ]; then
    args_add="--do_train --predict_type=dummy"
      elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    args_add="--do_eval --predict_type=${step}"
      fi

      python3 run_dst.py \
        --task_name=${TASK} \
        --data_dir=${DATA_DIR} \
        --dataset_config=dataset_config/${TASK}.json \
        --model_type="custombert" \
        --model_name_or_path="bert-base-uncased" \
        --do_lower_case \
        --learning_rate=1e-4 \
        --num_train_epochs=10 \
        --max_seq_length=180 \
        --per_gpu_train_batch_size=48\
        --per_gpu_eval_batch_size=1 \
        --output_dir=${OUT_DIR} \
        --save_epochs=2 \
        --logging_steps=10 \
        --warmup_proportion=0.1 \
        --adam_epsilon=1e-6 \
        --label_value_repetitions \
        --swap_utterances \
        --append_history \
        --use_history_labels \
        --delexicalize_sys_utts \
        --class_aux_feats_inform \
        --class_aux_feats_ds \
        --eval_all_checkpoints \
        --target=${target} \
        --enableInterEmb=0 \
        --enable_ds=0 \
        --max_slot_len=32 \
        --class_aux_feats_inform \
        --class_aux_feats_ds \
        --mode="normal" \
        ${args_add} \
        2>&1 | tee ${OUT_DIR}/${step}.log

      if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
        python3 metric_bert_dst.py \
          ${TASK} \
          dataset_config/${TASK}.json \
          "${OUT_DIR}/pred_res.${step}*json" \
          ${target} \
          ${step} 2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
      fi
  done
done
#--class_aux_feats_inform --class_aux_feats_ds
#_-class_aux_feats_inform \
