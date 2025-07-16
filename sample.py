import os
from contextlib import nullcontext

import sentencepiece as spm
import torch

from buggi import DOT, DOTConfig

init_from = "resume"
out_dir = "out"
model_path = "16768_full_txt.model"
start = "నమస్కారం, నేను తెలుగు"
num_samples = 3
max_new_tokens = 100
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "float16"
compile = False
batch_size = 1

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)

torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

print(f"Loading tokenizer from {model_path}...")
try:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size()}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit(1)

def encode(text):
    return tokenizer.encode_as_ids(text)

def decode(token_ids):
    return tokenizer.decode_pieces([tokenizer.id_to_piece(id) for id in token_ids])

if init_from == "resume":
    possible_checkpoints = [
        os.path.join(out_dir, "full_checkpoint.pth"),
        os.path.join(out_dir, "checkpoint.pth"),
        os.path.join(out_dir, "ckpt.pt"),
    ]

    ckpt_path = None
    for path in possible_checkpoints:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None:
        print(f"Error: No checkpoint found in {out_dir}")
        print("Available files:")
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                print(f"  {f}")
        exit(1)

    print(f"Loading checkpoint from {ckpt_path}...")

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print("Checkpoint loaded to CPU")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    if "model_args" in checkpoint:
        model_args = checkpoint["model_args"]
    elif "config" in checkpoint:
        config = checkpoint["config"]
        model_args = {
            "n_layer": config.get("n_layer", 16),
            "n_head": config.get("n_head", 16),
            "n_embd": config.get("n_embd", 1024),
            "block_size": config.get("block_size", 1024),
            "bias": config.get("bias", False),
            "dropout": config.get("dropout", 0.0),
            "vocab_size": config.get("vocab_size", tokenizer.vocab_size()),
        }
    else:
        model_args = {
            "n_layer": 16,
            "n_head": 16,
            "n_embd": 1024,
            "block_size": 1024,
            "bias": False,
            "dropout": 0.0,
            "vocab_size": tokenizer.vocab_size(),
        }

    print(f"Model config: {model_args}")

    try:
        model_config = DOTConfig(**model_args)
        model = DOT(model_config)
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        exit(1)

    try:
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            print("Error: No model state found in checkpoint")
            exit(1)

        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        print("Model weights loaded successfully!")

    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

elif init_from == "scratch":
    print("Error: Cannot sample from scratch model. Train the model first.")
    exit(1)

model.eval()
print(f"Moving model to {device}...")
model = model.to(device)

if device == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print(f"\nPrompt: {start}")
start_ids = encode(start)
print(f"Encoded prompt: {start_ids}")
print(f"Prompt length: {len(start_ids)} tokens")

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    for i in range(max_new_tokens):
        idx_cond = (
            idx
            if idx.size(1) <= model.config.block_size
            else idx[:, -model.config.block_size :]
        )

        with ctx:
            try:
                logits, _ = model(idx_cond)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM at token {i}. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    logits, _ = model(idx_cond)
                else:
                    raise e

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if i % 20 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    return idx[0].tolist()

print(f"\nGenerating {num_samples} samples with {max_new_tokens} tokens each...")
print("=" * 80)

for k in range(num_samples):
    print(f"\nSample {k+1}/{num_samples}:")
    print("-" * 50)

    try:
        generated_ids = generate(
            model, start_ids, max_new_tokens, temperature=temperature, top_k=top_k
        )
        generated_text = decode(generated_ids)
        print(f"Generated: {generated_text}")

        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error generating sample {k+1}: {e}")
        if device == "cuda":
            torch.cuda.empty_cache()
        continue

    print("-" * 50)

print("\n" + "=" * 80)
print("Sampling Complete!")

if device == "cuda":
    print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
