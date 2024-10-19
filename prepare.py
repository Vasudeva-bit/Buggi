import os

import numpy as np
import sentencepiece as spm

# make encode use all processor cores for efficiency

if __name__ == "__main__":
    # Get all files in the folder
    files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".utf8")]
    sp = spm.SentencePieceProcessor()
    sp.load("m.model")
    output = np.array([], dtype=np.int32)
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            text = file.read()
            ids = np.array(sp.encode_as_ids(text), dtype=np.int32)
            output = np.concatenate((output, ids))
    output.tofile("out/telugu.bin")
