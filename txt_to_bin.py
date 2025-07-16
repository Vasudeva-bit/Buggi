import argparse
import glob
import os

import numpy as np
import sentencepiece as spm


def text_to_bin(input_file, model_path, out_dir, shard_size=100000000):
    print(f"Loading tokenizer from {model_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(model_path)
    vocab_size = tokenizer.vocab_size()
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading text from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing text...")
    tokens = tokenizer.encode_as_ids(text)
    print(f"Total tokens: {len(tokens):,}")

    tokens_np = np.array(tokens, dtype=np.uint16)

    total_tokens = len(tokens_np)
    num_shards = (total_tokens + shard_size - 1) // shard_size
    print(f"Saving {num_shards} shards of max {shard_size:,} tokens each")

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_tokens)
        shard_tokens = tokens_np[start_idx:end_idx]

        shard_filename = os.path.join(out_dir, f"shard_{i:04d}.bin")
        shard_tokens.tofile(shard_filename)
        print(f"Saved shard {i}: {shard_filename} ({len(shard_tokens):,} tokens)")

    print(f"Conversion complete! Shards saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text to binary shards")
    parser.add_argument(
        "--input_file", default="telugu_training_data_full.txt", help="Input text file"
    )
    parser.add_argument(
        "--model_path", default="16768_full_txt.model", help="SentencePiece model path"
    )
    parser.add_argument(
        "--out_dir", default="data_shards", help="Output directory for shards"
    )
    parser.add_argument(
        "--shard_size", type=int, default=100000000, help="Tokens per shard"
    )

    args = parser.parse_args()

    text_to_bin(args.input_file, args.model_path, args.out_dir, args.shard_size)
