#! pip install tokenizers

# from pathlib import Path
# import torch

# from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# torch.cuda.is_available()

# paths = "data/hf_train.txt"
with open("C:\Users\gangu\Desktop\pytorch roberta\hf_train.txt") as ft:
    data_train = ft.read()
test_text = data_train.split("\n")
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files="C:\Users\gangu\Desktop\pytorch roberta\hf_train.txt", vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("TwitterRoBERTa")
tokenizer = ByteLevelBPETokenizer(
    "./TwitterRoBERTa/vocab.json",
    "./TwitterRoBERTa/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(
    tokenizer.encode("Mi estas Julien.").tokens
)