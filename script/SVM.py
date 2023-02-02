import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
import pickle
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from gensim.utils import tokenize
from gensim.models import Phrases, Word2Vec

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


class PreProc:
    def __init__(self):
        pass

    def remove_more_stop_words(self, words_arr):
        if len(words_arr) > 0:
            remove = np.vectorize(lambda word: self.word_counts[word] > 50)
            return words_arr[remove(words_arr)]
        else:
            return ["empty"]

    # return low frequency words + words that don't appear in train
    def remove_more_stop_words_test(self, words_arr):
        if len(words_arr) > 0:
            remove = np.vectorize(
                lambda word: False
                if not word in self.word_counts
                else (True if self.word_counts[word] > 50 else False)
            )
            return words_arr[remove(words_arr)]
        else:
            return ["empty"]

    # find the correct wordnet tag for nltk pos_tag equivalent
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    # lemmatize
    def lemmatize(self, word_list):
        pos_tag = nltk.pos_tag(list(word_list))
        words = [
            self.lemmatizer.lemmatize(i[0], self.get_wordnet_pos(i[1])) for i in pos_tag
        ]
        return words

    def preproc(self, df, is_test=False):
        feature_df = df.copy()
        # convert lower case
        feature_df["text"] = feature_df["text"].str.lower()
        # remove URLS
        feature_df["text"] = feature_df["text"].str.replace(
            r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
            "",
            regex=True,
        )
        # remove mentions
        feature_df["text"] = feature_df["text"].str.replace(
            r"@([a-zA-Z0-9_]{1,50})", "", regex=True
        )
        # remove hashtags
        feature_df["text"] = feature_df["text"].str.replace(
            r"#([a-zA-Z0-9_]{1,50})", "", regex=True
        )
        # remove punctuations
        feature_df["text"] = feature_df["text"].str.replace(r"[^\w\s]", "", regex=True)
        # tokenize
        feature_df["text"] = feature_df["text"].apply(
            lambda x: np.asarray(list(tokenize(x)))
        )

        if not is_test:
            # encode target labels
            label_encoder = LabelEncoder()
            label_encoder.fit(feature_df["target"])
            feature_df["target"] = label_encoder.transform(feature_df["target"])
            # check word counts
            self.word_counts = (
                feature_df.explode("text")["text"].value_counts().to_dict()
            )
            # remove low frequency words
            feature_df["text"] = feature_df["text"].apply(self.remove_more_stop_words)

        elif is_test:
            feature_df["text"] = feature_df["text"].apply(
                self.remove_more_stop_words_test
            )

        # lemmatize
        self.lemmatizer = WordNetLemmatizer()
        feature_df["text"] = feature_df["text"].apply(self.lemmatize)

        if not is_test:
            # vocab
            vocabulary = feature_df.explode("text")["text"].value_counts().to_dict()
            # vocab length
            vocab_length = len(vocabulary)
            return feature_df, self.word_counts, vocabulary, vocab_length

        elif is_test:
            return feature_df


path_df_SVM_train = (
    os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    + "\\Dataframe\\lemmatized_feature_df_SVM.pkl"
)
path_df_SVM_test = (
    os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    + "\\Dataframe\\lemmatized_feature_df_test_SVM.pkl"
)
# Uncomment and load dataframe if already pro-processed and saved
feature_df = pd.read_pickle(path_df_SVM_train)
feature_df_test = pd.read_pickle(path_df_SVM_test)


def tfidf(train_df, test_df):
    feature_df = train_df.copy()
    feature_df["text"] = feature_df["text"].str.join(" ")
    x_train_df, x_val_df = train_test_split(
        feature_df, stratify=feature_df["target"], random_state=40, test_size=0.2
    )
    v_c = TfidfVectorizer(max_features=90000)
    tfid_transformer = v_c.fit(x_train_df["text"])
    x_c = tfid_transformer.transform(x_train_df["text"])
    x_c_v = tfid_transformer.transform(x_val_df["text"])
    x_c = x_c.astype("float32")
    # x_c = x_c.toarray().astype('float32')
    x_c_v = x_c_v.astype("float32")
    # x_c_v = x_c_v.toarray().astype('float32')
    y_train = np.asarray(x_train_df["target"].tolist())
    y_val = np.asarray(x_val_df["target"].tolist())

    test_df["text"] = test_df["text"].str.join(" ")
    x_c_test = tfid_transformer.transform(test_df["text"])
    x_c_test = x_c_test.astype("float32")
    # x_c_test = x_c_test.todense().astype('float32')

    return x_c, x_c_v, y_train, y_val, x_c_test


x_c, x_c_v, y_train, y_val, x_c_test = tfidf(feature_df, feature_df_test)


def train_SVM(x, y):
    SVM_classifier_M = LinearSVC(random_state=40)
    model_SVM = SVM_classifier_M.fit(x, y)
    # path_models_SVM = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\\Models\\SVM_classifier_new.pkl'
    # with open(path_models_SVM, 'wb') as f:
    #    pickle.dump(model_SVM, f)
    return model_SVM


model_SVM = train_SVM(x_c, y_train)
