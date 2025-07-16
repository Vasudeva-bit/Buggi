#!/usr/bin/env python3
import argparse
import os

import sentencepiece as smp


def train_tokenizer_streaming(
    input_file, model_prefix, vocab_size, max_sentence_length=2048
):
    telugu_chars = "అఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళక్షజ్ఞ"

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return None

    num_threads = max(1, os.cpu_count() // 2)
    print(f"Using {num_threads} threads for streaming training.")
    print(f"Processing large file: {input_file}")

    file_size_gb = os.path.getsize(input_file) / (1024**3)
    print(f"Input file size: {file_size_gb:.2f} GB")

    command = (
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        f"--model_type=unigram --max_sentence_length={max_sentence_length} "
        f"--shuffle_input_sentence=true --character_coverage=0.995 "
        f"--num_threads={num_threads} --split_digits=true "
        f"--allow_whitespace_only_pieces=true --byte_fallback=true "
        f"--required_chars={telugu_chars} "
        f"--user_defined_symbols=<PAD>,<UNK>,<BOS>,<EOS>,<MASK> --unk_surface=⁇ "
        f"--normalization_rule_name=nmt_nfkc_cf "
        f"--seed_sentencepiece_size=2000000 "
        f"--shrinking_factor=0.75 --max_sentencepiece_length=16 "
        f"--split_by_unicode_script=true --split_by_whitespace=true "
        f"--split_by_number=true --treat_whitespace_as_suffix=false "
        f"--hard_vocab_limit=true --use_all_vocab=false "
        f"--input_sentence_size=2000000 "
        f"--mining_sentence_size=2000000 "
        f"--training_sentence_size=2000000"
    )

    try:
        print("Starting tokenizer training (this may take a while for large files)...")
        smp.SentencePieceTrainer.train(command)

        model_file = f"{model_prefix}.model"
        if os.path.exists(model_file):
            sp = smp.SentencePieceProcessor()
            sp.load(model_file)
            actual_vocab_size = sp.vocab_size()
            print(f"Tokenizer training completed successfully!")
            print(f"Model saved: {model_file}")
            print(f"Vocab saved: {model_prefix}.vocab")
            print(f"Requested vocab size: {vocab_size}")
            print(f"Actual vocab size: {actual_vocab_size}")
            return model_file
        else:
            print(f"Error: Model file {model_file} was not created!")
            return None

    except Exception as e:
        print(f"Tokenizer training failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer with streaming for large files."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the training data text file.",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        required=True,
        help="Prefix for the model and vocab files.",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=16768, help="Vocabulary size."
    )
    parser.add_argument(
        "--max_sentence_length", type=int, default=2048, help="Maximum sentence length."
    )

    args = parser.parse_args()

    print(f"Training tokenizer with streaming approach...")
    print(f"Input file: {args.input_file}")
    print(f"Model prefix: {args.model_prefix}")
    print(f"Vocab size: {args.vocab_size}")

    result = train_tokenizer_streaming(
        args.input_file, args.model_prefix, args.vocab_size, args.max_sentence_length
    )

    if result:
        print(f"Success! Tokenizer saved as {result}")
    else:
        print("Training failed!")
