#!/bin/bash
# EXPERIMENT: Loss Function Ablation
# Usage:
#   bash scripts/TimeLLM_Stock_DirectionalOnly.sh mse
#   bash scripts/TimeLLM_Stock_DirectionalOnly.sh directional 0.3
#   bash scripts/TimeLLM_Stock_DirectionalOnly.sh asymmetric 1.5 1.0
#   bash scripts/TimeLLM_Stock_DirectionalOnly.sh weighted_dir 0.5

loss_type=${1:-directional}
direction_weight=${2:-0.3}
up_penalty=${3:-1.5}
down_penalty=${4:-1.0}

# Set model_id and comment based on loss type
case $loss_type in
  mse)
    model_id="VCB_stock_mse"
    model_comment="Loss-MSE"
    ;;
  directional)
    model_id="VCB_stock_directional_dw${direction_weight}"
    model_comment="Loss-Directional-dw${direction_weight}"
    ;;
  asymmetric)
    model_id="VCB_stock_asymmetric_up${up_penalty}"
    model_comment="Loss-Asymmetric-up${up_penalty}-dn${down_penalty}"
    ;;
  weighted_dir)
    model_id="VCB_stock_weighted_dw${direction_weight}"
    model_comment="Loss-WeightedDir-dw${direction_weight}"
    ;;
  *)
    echo "Unknown loss_type: $loss_type"
    echo "Options: mse | directional | asymmetric | weighted_dir"
    exit 1
    ;;
esac

echo "============================================"
echo "EXPERIMENT: Loss Function Ablation"
echo "============================================"
echo "Loss Type:        $loss_type"
echo "Direction Weight: $direction_weight"
echo "Up Penalty:       $up_penalty"
echo "Down Penalty:     $down_penalty"
echo "Model ID:         $model_id"
echo "============================================"

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2029 run_stock_directional_only.py \
  --loss_type $loss_type \
  --direction_weight $direction_weight \
  --up_penalty $up_penalty \
  --down_penalty $down_penalty \
  --patching_mode single \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path vcb_stock_indicators_v2.csv \
  --model_id $model_id \
  --model TimeLLM \
  --data Stock \
  --features MS \
  --target "Adj Close" \
  --freq d \
  --seq_len 60 \
  --label_len 30 \
  --pred_len 1 \
  --enc_in 13 \
  --dec_in 13 \
  --c_out 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 6 \
  --train_epochs 10 \
  --patience 10 \
  --patch_len 8 \
  --stride 4 \
  --prompt_domain 1 \
  --model_comment "$model_comment"
