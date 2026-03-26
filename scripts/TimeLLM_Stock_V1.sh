#!/bin/bash

# Time-LLM Stock Prediction V1 - Dynamic Prompt + MSE baseline
# So sánh với V0: thêm dynamic per-sample prompts, giữ MSE thuần (direction_weight=0)

# Configuration
model_name=TimeLLM_Stock
prediction_type=${1:-short_term}
direction_weight=${2:-0}

echo "============================================"
echo "Time-LLM Stock Prediction V1"
echo "============================================"
echo "Prediction Type: $prediction_type"
echo "Direction Weight: $direction_weight"
echo ""
echo "Configuration:"
echo "  - Data: V2 data with momentum features"
echo "  - Features: 13 (RSI, MACD, BB, Volume, ROC + momentum_1d/3d/5d/10d + MA/Trend/Price/Vol)"
echo "  - Prompts: Dynamic per-sample (prompts_v2)"
echo "  - Loss: MSE (direction_weight=0 => MSE only)"
echo "============================================"

# Settings (giống V0 để so sánh công bằng)
llm_model=GPT2
llm_dim=768
llm_layers=6
d_model=32
d_ff=128
batch_size=16
learning_rate=0.001
dropout=0.1
train_epochs=10
patience=10
seq_len=60
label_len=30

# Set pred_len and prompt file based on prediction type
if [ "$prediction_type" == "short_term" ]; then
    pred_len=1
    data_path="vcb_stock_indicators_v2.csv"
    prompt_data_path="prompts_v2_short_term.json"
    model_id="VCB_v1_dynamic_60_1"
else
    pred_len=60
    data_path="vcb_stock_indicators_v2.csv"
    prompt_data_path="prompts_v2_mid_term.json"
    model_id="VCB_v1_dynamic_60_60"
    batch_size=8
    learning_rate=0.0005
fi

# V2 data: 13 input features (không tính Adj Close)
enc_in=13
dec_in=13
c_out=1

echo ""
echo "Model Configuration:"
echo "  - Input features: $enc_in"
echo "  - Sequence length: $seq_len"
echo "  - Prediction length: $pred_len"
echo "  - Data file: $data_path"
echo "  - Prompt file: $prompt_data_path"
echo ""

# Run training
python run_stock_training_v1.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id $model_id \
    --model $model_name \
    --data Stock \
    --root_path ./dataset/dataset/stock/ \
    --data_path $data_path \
    --prompt_data_path $prompt_data_path \
    --features MS \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --d_model $d_model \
    --d_ff $d_ff \
    --llm_model $llm_model \
    --llm_dim $llm_dim \
    --llm_layers $llm_layers \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --dropout $dropout \
    --train_epochs $train_epochs \
    --patience $patience \
    --prediction_type $prediction_type \
    --direction_weight $direction_weight \
    --use_dynamic_prompt \
    --model_comment "V1-DynamicPrompt-MSE"

echo ""
echo "============================================"
echo "V1 Training Complete!"
echo "============================================"
