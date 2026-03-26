#!/bin/bash
# TimeLLM Stock - FFT + Dynamic Prompt (MSE only, no directional loss)
# Kết hợp: FFT/Attention patching + dynamic per-sample prompts
# Base on: TimeLLM_Stock_FFT.sh
# Chỉ khác FFT: thêm --use_dynamic_prompt, giữ --direction_weight 0 (MSE thuần)
#
# Usage: bash scripts/TimeLLM_Stock_FFT_DynPrompt.sh [patching_mode]
# patching_mode: frequency_aware (default), multi_scale, single

patching_mode=${1:-frequency_aware}
prediction_type=${2:-short_term}

echo "============================================"
echo "TimeLLM Stock - FFT + Dynamic Prompt"
echo "============================================"
echo "Patching Mode: $patching_mode"
echo "Prediction Type: $prediction_type"
echo "Loss: MSE only (no directional)"
echo "Prompts: Dynamic per-sample"
echo "============================================"

# Data + prompt path theo prediction_type
if [ "$prediction_type" == "short_term" ]; then
    pred_len=1
    data_path="vcb_stock_indicators_v2.csv"
    prompt_data_path="prompts_v2_short_term.json"
    model_id="VCB_fft_dynprompt_60_1"
else
    pred_len=60
    data_path="vcb_stock_indicators_v2.csv"
    prompt_data_path="prompts_v2_mid_term.json"
    model_id="VCB_fft_dynprompt_60_60"
fi

echo ""
echo "Model Configuration:"
echo "  - Data: $data_path"
echo "  - Prompt: $prompt_data_path"
echo "  - Patching: $patching_mode"
echo "  - Pred len: $pred_len"
echo ""

accelerate launch --mixed_precision bf16 --num_processes 1 --main_process_port 2030 run_stock_fft_dynprompt.py \
  --use_dynamic_prompt \
  --patching_mode $patching_mode \
  --prompt_data_path $prompt_data_path \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/dataset/stock/ \
  --data_path $data_path \
  --model_id $model_id \
  --model TimeLLM \
  --data Stock \
  --features MS \
  --target "Adj Close" \
  --freq d \
  --seq_len 60 \
  --label_len 30 \
  --pred_len $pred_len \
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
  --train_epochs 30 \
  --patience 10 \
  --patch_len 8 \
  --stride 4 \
  --prompt_domain 1 \
  --model_comment "TimeLLM-$patching_mode-DynPrompt"

echo ""
echo "============================================"
echo "FFT + Dynamic Prompt Training Complete!"
echo "============================================"
