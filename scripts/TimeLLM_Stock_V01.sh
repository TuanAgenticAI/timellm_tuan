#!/bin/bash

# Time-LLM Stock Prediction V01 - Baseline + Dynamic Prompt
# Giống V0 hoàn toàn (data, model, MSE loss, hyperparams)
# Điểm khác DUY NHẤT so với V0: thêm dynamic per-sample prompts
#
# Mục đích: đo tác động riêng của dynamic prompts so với baseline V0

# Configuration
model_name=TimeLLM  # Giữ nguyên model gốc như V0
prediction_type=${1:-short_term}

echo "============================================"
echo "Time-LLM Stock Prediction V01"
echo "  = V0 Baseline + Dynamic Prompt"
echo "============================================"
echo "Prediction Type: $prediction_type"
echo "Model: Original TimeLLM (no enhancements)"
echo "Loss: Standard MSE (no directional component)"
echo "Prompts: Dynamic per-sample (V0 prompts)"
echo "============================================"

# Settings — giữ nguyên như V0 để so sánh công bằng
llm_model=GPT2
llm_dim=768
llm_layers=6
d_model=32
d_ff=128
batch_size=16
learning_rate=0.001
dropout=0.1
train_epochs=50
patience=10
seq_len=60
label_len=30

# Set pred_len + data + prompt theo prediction_type
if [ "$prediction_type" == "short_term" ]; then
    pred_len=1
    data_path="vcb_stock_indicators_v2.csv"          # V0 data (như V0)
    prompt_data_path="prompts_v2_short_term.json"     # V0 prompts
    model_id="VCB_v01_dynprompt_60_1"
else
    pred_len=60
    data_path="vcb_stock_indicators_v0.csv"           # V0 data (như V0)
    prompt_data_path="prompts_v0_mid_term.json"        # V0 prompts
    model_id="VCB_v01_dynprompt_60_60"
    batch_size=8
    learning_rate=0.0005
fi

# V0 data có 6 cột số: RSI, MACD, BB_Position, Volume_Norm, ROC, Adj Close
enc_in=13
dec_in=13
c_out=1

echo ""
echo "Model Configuration:"
echo "  - Input features: $enc_in (RSI, MACD, BB_Position, Volume_Norm, ROC, Adj Close)"
echo "  - Sequence length: $seq_len"
echo "  - Prediction length: $pred_len"
echo "  - Data file: $data_path"
echo "  - Prompt file: $prompt_data_path (dynamic per-sample)"
echo ""

# Run training
python run_stock_training_v01.py \
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
    --use_dynamic_prompt \
    --model_comment "V01-Baseline-DynPrompt"

echo ""
echo "============================================"
echo "V01 Training Complete!"
echo "============================================"
