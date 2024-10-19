import math
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model import DOT, DOTConfig

out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"
# data
data_dir = "data"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
# understand warmup and cosine scheduler clearly
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iter = 600000
min_lr = 6e-5
backend = "nccl"
# system
device = "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# exec(open("configurator.py").read()) :revert
config = {k: globals()[k] for k in config_keys}
# verfiy config and it's use in nanoGPT

# revert commented code
ddp = 0  # int(os.environ.get("RANK", -1)) != -1
ddp_rank = 0  # int(os.environ["RANK"])
assert ddp_rank == 0  # int(os.environ["LOCAL_RANK"])
ddp_world_size = 0  # int(os.environ["WORLD_SIZE"])
if not ddp:
    print("Running in non-ddp mode")
    # sys.exit(0)


master_process = ddp == 0
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * block_size * block_size
print(f"tokens per iter: {tokens_per_iter}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

seed_offset = ddp_rank
torch.manual_seed(42 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if ddp else "cpu"
ptdtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype) if ddp else nullcontext()
)
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))


# get batchs from telugu.bin
class Data(Dataset):
    def __init__(self, bin_file_path, block_size):
        self.bin_file_path = bin_file_path
        self.block_size = block_size
        self.data = np.memmap(bin_file_path, dtype=np.uint16, mode="r")

    def __getitem__(self, index):
        x = self.data[index : index + self.block_size].astype(np.int64)
        y = self.data[index + 1 : index + self.block_size].astype(np.int64)
        y = np.append(y, -1)  # append eot token
        return x, y

    def __len__(self):
        return len(self.data) - self.block_size - 1


dataset = Data(os.path.join(out_dir, "telugu.bin"), block_size)
# what is model here? it isn't initialized yet


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iter:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iter - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def create_ddp_model(model, rank):
    init_process_group(backend=backend)
    model = model.to(rank)
    ddp_model = DDP(model, rank)
    return ddp_model


def create_dataloader(bin_dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(
        bin_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    data_loader = DataLoader(bin_dataset, batch_size=batch_size, sampler=sampler)
    return data_loader


def cleanup():
    """Cleans up the distributed backend"""
    dist.destroy_process_group()


def train(rank, world_size, bin_dataset, model, optimizer, epochs, batch_size):
    data_loader = create_dataloader(
        bin_dataset, batch_size, rank, world_size, drop_last=True
    )
    model = create_ddp_model(model, rank)
    epochs = max_iters // len(data_loader)  # just to balance loader and iters
    # Training loop
    for epoch in range(epochs):
        # set lr using get_lr(epoch)
        data_loader.sampler.set_epoch(epoch)  # Ensure each epoch loads different data
        # adapt it to gradient accumulation :to_do
        # verify no of batchs inside data_loader
        for batch in data_loader:
            X, y = batch.to(device, non_blocking=True)
            with ctx:
                _, loss = model(X, labels=y)
            # backprop
            scaler.scale(loss).backward()
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "loss": loss,
        # also save config
    }
    torch.save(checkpoint, os.path.join(out_dir, "checkpoint.pth"))
    cleanup()


model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    vocab_size=DOTConfig.vocab_size,
)

config = DOTConfig(**model_args)
model = DOT(config)
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if compile:
    print("Compiling model")
    unoptimized_model = model
    model = torch.compile(unoptimized_model)

train(ddp_rank, ddp_world_size, dataset, model, optimizer, 1, batch_size)

# 1. grad accum
# 2. evaluation inside the function, no need of seperate function to estimate loss
# 3. test train loop and proper grad updates
