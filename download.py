from datasets import load_dataset
ds = load_dataset("uonlp/CulturaX",
                  "te",
                  token="") # use your token on HF, under access tokens; not mine haha.
ds.save_to_disk("./data")
