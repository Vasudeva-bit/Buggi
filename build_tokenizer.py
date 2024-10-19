import os

import sentencepiece as spm

folder_path = "data"

# Get all files in the folder
files = [
    os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".utf8")
]

# Create a comma-separated string of file paths
input_files = ",".join(files)
# change vocab size to 16767 + 1 for eot | refactor
spm.SentencePieceTrainer.train(
    f"--input={input_files} --model_prefix=m --vocab_size=6400 --model_type=bpe"
)
