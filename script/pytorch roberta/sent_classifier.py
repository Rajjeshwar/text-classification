from transformers import RobertaTokenizerFast
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

tokenizer = RobertaTokenizerFast.from_pretrained("C:\Users\gangu\Desktop\pytorch roberta\TwitterRoBERTa", max_len=512)

with open("/misc/data22-brs/IVVES01/stage_nlp/datachung/sentiment_analysis/hf_train.txt") as ft:
    data_train = ft.read()
    data_train = data_train.strip()
train_text = data_train.split("\n")

results_data = pd.read_csv('C:\Users\gangu\Desktop\pytorch roberta\TwitterRoBERTa/train_results.csv')
results_data.loc[results_data['target'] == "negative", 'labels'] = 0
results_data.loc[results_data['target'] == "positive", 'labels'] = 2
results_data.loc[results_data['target'] == "neutral", 'labels'] = 1
train_labels = list(results_data['labels'].astype(int))

# with open("/misc/data22-brs/IVVES01/stage_nlp/datachung/sentiment_analysis/hf_test.txt") as ft:
#     data_train = ft.read()
# test_text = data_train.split("\n")

train_encoding = tokenizer(train_text, truncation=True, padding=True)
# test_encoding = tokenizer(test_text, truncation=True, padding=True)


import torch


class TweetDataset(torch.utils.data.Dataset):
    """
    Class to store the tweet data as PyTorch Dataset
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # an encoding can have keys such as input_ids and attention_mask
        # item is a dictionary which has the same keys as the encoding has
        # and the values are the idxth value of the corresponding key (in PyTorch's tensor format)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TweetDataset(train_encoding, train_labels)

training_args = TrainingArguments(
    output_dir='/misc/data22-brs/IVVES01/stage_nlp/datachung/sentiment_analysis/results_2',          # output directory
    num_train_epochs=9,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='C:\Users\gangu\Desktop\pytorch roberta\TwitterRoBERTa/logs',            # directory for storing logs
    logging_steps=10,
    # load_best_model_at_end=True,
)



# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
id2label = {0: "negative", 1: "positive", 2: "neutral"}
label2id = {"negative": 0, "positive": 1, "neutral": 2}
model = RobertaForSequenceClassification.from_pretrained("C:\Users\gangu\Desktop\pytorch roberta\TwitterRoBERTa",
                                                        num_labels=3, id2label=id2label, label2id=label2id)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    # eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()