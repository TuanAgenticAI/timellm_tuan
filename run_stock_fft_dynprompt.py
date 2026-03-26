"""
Stock Training - FFT Patching + Dynamic Prompt (MSE only)
==========================================================
Kết hợp:
  1. FFT/Attention patching  (--patching_mode: frequency_aware / multi_scale / single)
  2. Dynamic per-sample prompts (--use_dynamic_prompt)
  3. Standard MSE loss  (không directional)

Mục đích: tách biệt tác động của FFT patching + dynamic prompt
so với baseline (V0) và các ablation khác.

Usage:
    python run_stock_fft_dynprompt.py \
        --patching_mode frequency_aware \
        --use_dynamic_prompt \
        --data_path vcb_stock_indicators_v0.csv \
        --prompt_data_path prompts_v0_short_term.json
"""

import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
import json

from models import TimeLLM
from data_provider.data_factory import data_provider

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, load_content


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_winrate(pred, true, prev_values):
    """Tính winrate (directional accuracy): tỉ lệ dự đoán đúng chiều tăng/giảm."""
    pred_up = pred[:, 0, 0] > prev_values
    true_up = true[:, 0, 0] > prev_values
    correct = (pred_up == true_up).sum().item()
    total = pred_up.numel()
    return correct, total


def vali(args, accelerator, model, vali_loader, criterion, mae_metric):
    """Validation: trả về MSE, MAE, Winrate."""
    total_mse, total_mae = [], []
    total_correct, total_samples = 0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(vali_loader, desc="Validating",
                          disable=not accelerator.is_local_main_process):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
            prompts = batch[4] if len(batch) == 5 else None

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp], dim=1
            ).float().to(accelerator.device)

            if prompts is not None:
                unwrapped = accelerator.unwrap_model(model)
                outputs = unwrapped(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    dynamic_prompts=prompts
                )
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            prev_values = batch_x[:, -1, f_dim].to(accelerator.device)

            pred, true = outputs.detach(), batch_y.detach()

            total_mse.append(criterion(pred, true).item())
            total_mae.append(mae_metric(pred, true).item())

            correct, total = compute_winrate(pred, true, prev_values)
            total_correct += correct
            total_samples += total

    model.train()
    avg_mse = np.average(total_mse)
    avg_mae = np.average(total_mae)
    winrate = total_correct / total_samples * 100 if total_samples > 0 else 0
    return avg_mse, avg_mae, winrate


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Time-LLM Stock - FFT Patching + Dynamic Prompt (MSE only)'
    )

    # Patching
    parser.add_argument('--patching_mode', type=str, default='frequency_aware',
                        choices=['frequency_aware', 'multi_scale', 'single'],
                        help='frequency_aware: FFT+attention | multi_scale: fixed | single: original')

    # Dynamic prompt
    parser.add_argument('--use_dynamic_prompt', action='store_true', default=False)
    parser.add_argument('--prompt_data_path', type=str, default=None,
                        help='Tên file JSON prompt (trong root_path)')

    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='VCB_fft_dynprompt_60_1')
    parser.add_argument('--model_comment', type=str, default='FFT-DynPrompt')
    parser.add_argument('--model', type=str, default='TimeLLM')
    parser.add_argument('--seed', type=int, default=2021)

    # Data config
    parser.add_argument('--data', type=str, default='Stock')
    parser.add_argument('--root_path', type=str, default='./dataset/dataset/stock/')
    parser.add_argument('--data_path', type=str, default='vcb_stock_indicators_v0.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='Adj Close')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--loader', type=str, default='modal')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # Forecast config
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--label_len', type=int, default=30)
    parser.add_argument('--pred_len', type=int, default=1)

    # Model config
    parser.add_argument('--enc_in', type=int, default=6)
    parser.add_argument('--dec_in', type=int, default=6)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--prompt_domain', type=int, default=1)
    parser.add_argument('--llm_model', type=str, default='GPT2')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)

    # Training config
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()

    # Auto prompt file nếu chưa truyền
    if args.prompt_data_path is None:
        data_lower = args.data_path.lower()
        is_v2 = '_v2' in data_lower
        if args.pred_len == 1:
            args.prompt_data_path = (
                'prompts_v2_short_term.json' if is_v2 else 'prompts_v0_short_term.json'
            )
        else:
            args.prompt_data_path = (
                'prompts_v2_mid_term.json' if is_v2 else 'prompts_v0_mid_term.json'
            )

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    setting = '{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_pm{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.model_comment,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.d_ff, args.patching_mode
    )

    args.content = load_content(args)

    accelerator.print(f"\n{'='*65}")
    accelerator.print(f"Time-LLM Stock - FFT Patching + Dynamic Prompt")
    accelerator.print(f"{'='*65}")
    accelerator.print(f"  Patching Mode : {args.patching_mode}")
    accelerator.print(f"  Dynamic Prompt: {args.use_dynamic_prompt} "
                      f"({args.prompt_data_path if args.use_dynamic_prompt else 'static only'})")
    accelerator.print(f"  Loss          : Standard MSE")
    accelerator.print(f"  Data          : {args.data_path} (enc_in={args.enc_in})")
    accelerator.print(f"  seq_len={args.seq_len}  pred_len={args.pred_len}")
    accelerator.print(f"{'='*65}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data, train_loader = data_provider(
        args, 'train', with_prompt=args.use_dynamic_prompt)
    vali_data, vali_loader = data_provider(
        args, 'val', with_prompt=args.use_dynamic_prompt)
    test_data, test_loader = data_provider(
        args, 'test', with_prompt=args.use_dynamic_prompt)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TimeLLM.Model(args).float()
    accelerator.print(
        f"Model: {args.model} | Patching: {args.patching_mode} | enc_in: {args.enc_in}"
    )
    accelerator.print(
        f"Train: {len(train_data)} | Val: {len(vali_data)} | Test: {len(test_data)}\n"
    )

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    scheduler = lr_scheduler.OneCycleLR(
        optimizer=model_optim,
        steps_per_epoch=train_steps,
        pct_start=args.pct_start,
        epochs=args.train_epochs,
        max_lr=args.learning_rate,
    )

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = \
        accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    history = {
        'train_mse': [], 'train_winrate': [],
        'vali_mse': [], 'vali_mae': [], 'vali_winrate': [],
        'test_mse': [], 'test_mae': [], 'test_winrate': []
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.train_epochs):
        train_mse = []
        train_correct, train_total = 0, 0

        model.train()
        epoch_time = time.time()

        for i, batch in tqdm(enumerate(train_loader),
                              total=len(train_loader),
                              desc=f"Epoch {epoch + 1}",
                              disable=not accelerator.is_local_main_process):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
            prompts = batch[4] if len(batch) == 5 else None

            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(
                batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp], dim=1
            ).float().to(accelerator.device)

            if prompts is not None:
                unwrapped = accelerator.unwrap_model(model)
                outputs = unwrapped(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    dynamic_prompts=prompts
                )
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            target = batch_y[:, -args.pred_len:, f_dim:]
            prev_values = batch_x[:, -1, f_dim]

            loss = criterion(outputs, target)
            train_mse.append(loss.item())

            # Winrate tracking
            correct, total = compute_winrate(
                outputs.detach(), target.detach(), prev_values.detach()
            )
            train_correct += correct
            train_total += total

            accelerator.backward(loss)
            model_optim.step()
            scheduler.step()

        train_mse_avg = np.average(train_mse)
        train_winrate = train_correct / train_total * 100 if train_total > 0 else 0

        # ── Validation & Test ─────────────────────────────────────────────────
        vali_mse, vali_mae, vali_winrate = vali(
            args, accelerator, model, vali_loader, criterion, mae_metric)
        test_mse, test_mae, test_winrate = vali(
            args, accelerator, model, test_loader, criterion, mae_metric)

        accelerator.print(
            f"Epoch {epoch + 1} | Time: {time.time() - epoch_time:.2f}s | "
            f"Train MSE: {train_mse_avg:.6f} | Train Winrate: {train_winrate:.2f}%"
        )
        accelerator.print(
            f"         | Vali MSE: {vali_mse:.6f} | Vali MAE: {vali_mae:.6f} "
            f"| Vali Winrate: {vali_winrate:.2f}%"
        )
        accelerator.print(
            f"         | Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f} "
            f"| Test Winrate: {test_winrate:.2f}%"
        )

        # Print FFT patch info
        if accelerator.is_local_main_process:
            unwrapped = accelerator.unwrap_model(model)
            if hasattr(unwrapped, 'print_patch_info'):
                unwrapped.print_patch_info(epoch=epoch + 1)

        history['train_mse'].append(float(train_mse_avg))
        history['train_winrate'].append(float(train_winrate))
        history['vali_mse'].append(float(vali_mse))
        history['vali_mae'].append(float(vali_mae))
        history['vali_winrate'].append(float(vali_winrate))
        history['test_mse'].append(float(test_mse))
        history['test_mae'].append(float(test_mae))
        history['test_winrate'].append(float(test_winrate))

        early_stopping(vali_mse, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        adjust_learning_rate(
            accelerator, model_optim, scheduler, epoch + 1, args, printout=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    if accelerator.is_local_main_process:
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        config_dict = {k: v for k, v in vars(args).items() if k != 'content'}
        config_dict['description'] = 'FFT Patching + Dynamic Prompt + MSE only'
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)

    accelerator.wait_for_everyone()

    accelerator.print(f"\n{'='*65}")
    accelerator.print(f"Training Completed!")
    accelerator.print(f"Patching: {args.patching_mode} | "
                      f"Dynamic Prompt: {args.use_dynamic_prompt}")
    accelerator.print(f"Model saved to: {path}")
    accelerator.print(f"{'='*65}")

    return path, history


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
