import pickle
import numpy as np
from keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from keras.layers import (
    Input,
    Activation,
    Dense,
    Permute,
    Dropout,
    add,
    dot,
    concatenate,
    LSTM,
)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tf.kears.layers import TextVectorization


class SimpleChatBot:
    def __init__(self):
        """
        Length of train data is 10.000
        Sample train data:
        (['Mary', 'moved', 'to', 'the', 'bathroom', '.', 'Sandra', 'journeyed', 'to', 'the', 'bedroom'], -> Story
        ['Is', 'Sandra', 'in', 'the', 'hallway', '?'], -> Question
        'no') -> Answer
        """
        self.tokenizer = None
        self.vocab = set()

        self.train_story_text: list = []
        self.train_question_text: list = []
        self.train_answers: list = []
        self.all_data: list = []
        self.train_story_seq: list[list] = []

        self.max_story_length: int = 0
        self.max_question_length: int = 0

        self.vocab_len: int
        with open(
            "app/data/Advanced-Chatbots-with-Deep-Learning-Python/Data/train_qa.txt",
            "rb",
        ) as f:
            self.train_data = pickle.load(f)
        # Length of test data is 1.000
        with open(
            "app/data/Advanced-Chatbots-with-Deep-Learning-Python/Data/test_qa.txt",
            "rb",
        ) as f:
            self.test_data = pickle.load(f)
        self.all_data = self.test_data + self.train_data

    def make_vocabulary(self) -> None:
        for story, question, answer in self.all_data:
            self.vocab = self.vocab.union(set(story))
            self.vocab = self.vocab.union(set(question))
        # add 'yes' and 'no' answers to the vocab
        self.vocab.add("yes")
        self.vocab.add("no")
        # get vocabulary length
        vocab_length = len(self.vocab) + 1

    def max_story_question_len(self):
        self.max_story_length = max([len(data[0]) for data in self.all_data])
        self.max_question_length = max([len(data[1]) for data in self.all_data])

    def create_tokenizer_sequence(self) -> None:
        # create tokenizer with no filter
        self.tokenizer = Tokenizer(filters=[])
        self.tokenizer.fit_on_texts(self.vocab)
        # we can see word_index of the tokenizer
        # it's like {'bathroom':1, 'in': 2, 'grabbed': 3, 'no': 18, 'yes': 31, ...}
        for story, question, answer in self.train_data:
            self.train_story_text.append(story)
            self.train_question_text.append(question)
            self.train_answers.append(answer)
        self.train_story_seq = self.tokenizer.texts_to_sequences(self.train_story_text)

    def vectorize_stories(self, data):
        X, Xq, Y = []
        for story, query, answer in data:
            x_story_indices = [
                self.tokenizer.word_index[word.lower()] for word in story
            ]
            x_query_indices = [
                self.tokenizer.word_index[word.lower()] for word in query
            ]
            y = np.zeros(len(self.tokenizer.word_index) + 1)
            y[self.tokenizer.word_index[answer]] = 1

            X.append(x_story_indices)
            Xq.append(x_query_indices)
            Y.append(y)
        return (
            pad_sequences(X, maxlen=self.max_story_length),
            pad_sequences(Xq, maxlen=self.max_question_length),
            np.array(Y),
        )
