#!/bin/bash

# Stock baseline (MSE only) using LSTM

model_name=LSTM

master_port=2028
num_process=1

seq_len=60
label_len=30
pred_type=${1:-short_term}  # short_term | mid_term

# Stock v2 (13 features) matches existing stock experiments
root_path=./dataset/dataset/stock/
data_path=vcb_stock_indicators_v2.csv

model_id_base="VCB_stock_lstm_baseline"
model_comment="Baseline-MSE-LSTM"

enc_in=13
dec_in=13
c_out=1

d_model=32
d_ff=128
n_heads=8
e_layers=2
d_layers=1
dropout=0.1

patch_len=8
stride=4

train_epochs=10
patience=10
batch_size=16
learning_rate=0.001

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_stock_baseline_mse.py \
  --prediction_type $pred_type \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id_base \
  --model $model_name \
  --data Stock \
  --features MS \
  --target "Adj Close" \
  --freq d \
  --seq_len $seq_len \
  --label_len $label_len \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --d_model $d_model \
  --n_heads $n_heads \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --d_ff $d_ff \
  --dropout $dropout \
  --patch_len $patch_len \
  --stride $stride \
  --patching_mode single \
  --prompt_domain 1 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --model_comment "$model_comment"

