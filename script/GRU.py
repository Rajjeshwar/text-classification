import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
import pickle


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    TfidfTransformer,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from gensim.utils import tokenize
from gensim.models import Phrases, Word2Vec

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    LSTM,
    SpatialDropout1D,
    GRU,
    Bidirectional,
    Conv1D,
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


# word2vec train and fit function
def create_embedding(df):
    sentence_list = [row for row in feature_df["text"]]
    w2v = Word2Vec(sample=3e-5, min_count=1)
    w2v.build_vocab(corpus_iterable=sentence_list)
    w2v.train(
        corpus_iterable=sentence_list,
        total_examples=w2v.corpus_count,
        epochs=100,
        report_delay=1,
    )
    key_index = w2v.wv.key_to_index
    # MAKE A DICT
    word_dict = {word: w2v.wv[word] for word in key_index}

    embedding_weights = w2v.wv.vectors
    # add zeros vector to align indices of keras and gensim tokenizer
    row_zero = np.zeros((100))
    embedding_weights = np.insert(embedding_weights, 0, row_zero, axis=0)
    return word_dict, embedding_weights


def to_sequences(train_df, test_df):
    # list of input sentences
    X = train_df["text"].values

    # array of 0,1 and 2 labels
    Y = train_df["target"].to_numpy()
    input_tokenizer = Tokenizer()
    input_tokenizer.fit_on_texts(X)
    encoded_X = input_tokenizer.texts_to_sequences(X)

    # right pad for equal length
    X_train = pad_sequences(encoded_X, maxlen=50, padding="post")

    # generate train-val set
    x_train, x_val, y_train, y_val = train_test_split(
        X_train, Y, stratify=Y, test_size=0.2, random_state=40
    )

    # test sequences
    X_test = test_df["text"].values
    x_test_sequences = input_tokenizer.texts_to_sequences(X_test)
    x_test = pad_sequences(x_test_sequences, maxlen=50, padding="post")
    return x_train, x_val, x_test, y_train, y_val


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


path_df_NN_train = (
    os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    + "\\Dataframe\\lemmatized_feature_df_LSTM.pkl"
)
path_df_NN_test = (
    os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    + "\\Dataframe\\lemmatized_feature_df_test_LSTM.pkl"
)
feature_df = pd.read_pickle(path_df_NN_train)
feature_df_test = pd.read_pickle(path_df_NN_test)

# using skipgra eembeddings from lstm
path_word_dict = (
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\\Models\\word_dict.pkl"
)
path_embedding_weights = (
    os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    + "\\Models\\Word2Vec_weights.pkl"
)

word_dict = pickle.load(open(path_word_dict, "rb"))
embedding_weights = pickle.load(open(path_embedding_weights, "rb"))

x_train, x_val, x_test, y_train, y_val = to_sequences(feature_df, feature_df_test)


def create_model_GRU():
    model = Sequential()
    model.add(
        Embedding(
            input_dim=embedding_weights.shape[0],
            output_dim=embedding_weights.shape[1],
            weights=[embedding_weights],
            trainable=False,
            input_length=50,
        )
    )
    model.add(SpatialDropout1D(0.3))
    model.add(GRU(256, dropout=0.5))
    model.add(Dense(3, activation="softmax"))
    return model


GRU_model = create_model_GRU()


def train(x, y, model, optimizer, loss, metrics, batch_size, epochs, validation_data):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        shuffle=True,
    )
    return history


model_metrics = train(
    x_train,
    y_train,
    model=GRU_model,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    batch_size=1024,
    epochs=50,
    validation_data=(x_val, y_val),
)
