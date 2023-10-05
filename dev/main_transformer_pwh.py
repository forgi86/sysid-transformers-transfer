from pathlib import Path
import time
import torch
import numpy as np
import math
from functools import partial
from dataset import PWHDataset
from torch.utils.data import DataLoader
from model import GPTConfig, GPT, warmup_cosine_lr
import tqdm
import argparse
import wandb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta system identification with transformers')

    # Overall
    parser.add_argument('--model-dir', type=str, default="out", metavar='S',
                        help='Saved model folder')
    parser.add_argument('--out-file', type=str, default="ckpt_out_big_pwh", metavar='S',
                        help='Saved model name')
    parser.add_argument('--in-file', type=str, default="ckpt_out_big", metavar='S',
                        help='Loaded model name (when resuming)')
    parser.add_argument('--init-from', type=str, default="pretrained", metavar='S',
                        help='Init from (scratch|resume|pretrained)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Seed for random number generation')

    # Dataset
    parser.add_argument('--nx', type=int, default=100, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--nu', type=int, default=1, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--ny', type=int, default=1, metavar='N',
                        help='model order (default: 5)')
    parser.add_argument('--seq-len', type=int, default=1024, metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--mag_range', type=tuple, default=(0.5, 0.97), metavar='N',
                        help='sequence length (default: 600)')
    parser.add_argument('--phase_range', type=tuple, default=(0.0, math.pi/2), metavar='N',
                        help='sequence length (default: 600)')

    # Model
    parser.add_argument('--n-layer', type=int, default=12, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-head', type=int, default=12, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--n-embd', type=int, default=768, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='bias in model')

    # Training
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--max-iters', type=int, default=1_000_000, metavar='N',
                        help='number of iterations (default: 1M)')
    parser.add_argument('--warmup-iters', type=int, default=10_000, metavar='N',
                        help='number of iterations (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='D',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--eval-interval', type=int, default=2000, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--eval-iters', type=int, default=100, metavar='N',
                        help='batch size (default:32)')
    parser.add_argument('--fixed-lr', action='store_true', default=False,
                        help='disables CUDA training')

    # Compute
    parser.add_argument('--threads', type=int, default=10,
                        help='number of CPU threads (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda-device', type=str, default="cuda:0", metavar='S',
                        help='cuda device (default: "cuda:0")')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='disables CUDA training')

    cfg = parser.parse_args()

    # Other settings
    cfg.log_wandb = True
    cfg.beta1 = 0.9
    cfg.beta2 = 0.95

    # Derived settings
    cfg.block_size = cfg.seq_len
    cfg.lr_decay_iters = cfg.max_iters
    cfg.min_lr = cfg.lr/10.0  #
    cfg.decay_lr = not cfg.fixed_lr
    cfg.eval_batch_size = cfg.batch_size

    # Init wandb
    if cfg.log_wandb:
        wandb.init(
            project="sysid-meta",
            #name="run1",
            # track hyperparameters and run metadata
            config=vars(cfg)
        )

    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create out dir
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(exist_ok=True)

    # Configure compute
    torch.set_num_threads(cfg.threads)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device_name = cfg.cuda_device if use_cuda else "cpu"
    device = torch.device(device_name)
    device_type = 'cuda' if 'cuda' in device_name else 'cpu'
    torch.set_float32_matmul_precision("high")

    # Create data loader
    train_ds = PWHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len, mag_range=cfg.mag_range,
                         phase_range=cfg.phase_range)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.threads)

    val_ds = PWHDataset(nx=cfg.nx, nu=cfg.nu, ny=cfg.ny, seq_len=cfg.seq_len, mag_range=cfg.mag_range,
                       phase_range=cfg.phase_range)
    val_dl = DataLoader(val_ds, batch_size=cfg.eval_batch_size, num_workers=cfg.threads)

    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, n_y=1, n_u=1, block_size=cfg.block_size,
                      bias=cfg.bias, dropout=cfg.dropout)  # start with model_args from command line

    if cfg.init_from == "scratch":
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif cfg.init_from == "resume" or cfg.init_from == "pretrained":
        ckpt_path = model_dir / f"{cfg.in_file}.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    model.to(device)

    if cfg.compile:
        model = torch.compile(model)  # requires PyTorch 2.0

    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
    if cfg.init_from == "resume":
        optimizer.load_state_dict(checkpoint['optimizer'])

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        loss = 0.0
        for eval_iter, (batch_y, batch_u) in enumerate(val_dl):
            if device_type == "cuda":
                batch_y = batch_y.pin_memory().to(device, non_blocking=True)
                batch_u = batch_u.pin_memory().to(device, non_blocking=True)
            _, loss_iter = model(batch_u, batch_y)
            loss += loss_iter.item()
            if eval_iter == cfg.eval_iters:
                break
        loss /= cfg.eval_iters
        model.train()
        return loss

    # Training loop
    LOSS_ITR = []
    LOSS_VAL = []
    SKIP_ITR = []
    loss_val = np.nan

    if cfg.init_from == "scratch" or cfg.init_from == "pretrained":
        iter_num = 0
        best_val_loss = np.inf
    elif cfg.init_from == "resume":
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint['best_val_loss']

    get_lr = partial(warmup_cosine_lr, lr=cfg.lr, min_lr=cfg.min_lr,
                     warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
    time_start = time.time()
    for iter_num, (batch_y, batch_u) in tqdm.tqdm(enumerate(train_dl, start=iter_num)):

        if torch.isnan(batch_y).any():
            print("Nan at iteration {iter_num}, skipping...") # wtf?
            SKIP_ITR.append(iter_num)
            continue

        if (iter_num % cfg.eval_interval == 0):# and iter_num > 0:
            loss_val = estimate_loss()
            LOSS_VAL.append(loss_val)
            print(f"\n{iter_num=} {loss_val=:.4f}\n")
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'train_time': time.time() - time_start,
                    'LOSS': LOSS_ITR,
                    'LOSS_VAL': LOSS_VAL,
                    'best_val_loss': best_val_loss,
                    'cfg': cfg,
                }
                torch.save(checkpoint, model_dir / f"{cfg.out_file}.pt")
        # determine and set the learning rate for this iteration
        if cfg.decay_lr:
            lr_iter = get_lr(iter_num)
        else:
            lr_iter = cfg.lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_iter

        if device_type == "cuda":
            batch_y = batch_y.pin_memory().to(device, non_blocking=True)
            batch_u = batch_u.pin_memory().to(device, non_blocking=True)
        batch_y_pred, loss = model(batch_u, batch_y)
        LOSS_ITR.append(loss.item())
        if iter_num % 100 == 0:
            print(f"\n{iter_num=} {loss=:.4f} {loss_val=:.4f} {lr_iter=}\n")
            if cfg.log_wandb:
                wandb.log({"loss": loss, "loss_val": loss_val})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter_num == cfg.max_iters-1:
            break

    time_loop = time.time() - time_start
    print(f"\n{time_loop=:.2f} seconds.")

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'train_time': time.time() - time_start,
        'LOSS': LOSS_ITR,
        'LOSS_VAL': LOSS_VAL,
        'best_val_loss': best_val_loss,
        'cfg': cfg,
    }
    torch.save(checkpoint, model_dir / f"{cfg.out_file}_last.pt")

    if cfg.log_wandb:
        wandb.finish()


