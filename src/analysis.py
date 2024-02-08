from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import time
import os

import config

class Analysis:

    def __init__(self, models_path, model_from_file, clean_meth, split_size):
        """
        Class Analysis handles running the topic modeling analysis with the BERTopic module.

        Parameters:
            models_path (str): Path where models are stored or retrieved
            model_from_file (bool): Use model saved in file or not
            clean_meth (str): Text cleaning method

        Attributes:
            model_file_name (str): file name for BERTopic object stored in file.
            bert_model (str): Pre-trained sentence BERT model name, defined in config.
            merged (bool): Indicator to show whether topic output has been merged or not.
        """

        self.models_path = models_path
        self.model_from_file = model_from_file

        self.clean_meth = clean_meth
        self.split_size = split_size
        self.chunk_size = config.parameters["chunk_size"]
        self.bert_model = config.parameters["bert_model"]

        if self.split_size == "chunk":
            self.model_file_name = f"bertopic_model_{self.bert_model}_{self.split_size}{self.chunk_size}_{self.clean_meth}.pkl"
        else:
            self.model_file_name = f"bertopic_model_{self.bert_model}_{self.split_size}_{self.clean_meth}.pkl"

        self.merged = False

    def initialize_topic_model(self, data):
        if self.model_from_file:
            print(f"Topic-model name: {self.model_file_name}. Reading topic-model from file...")
            start = time.time()
            topic_model = BERTopic.load(os.path.join(self.models_path, self.model_file_name))
            print(f"Time elapsed to load topic-model: {time.time() - start:.3f} seconds")
            return topic_model

        else:
            print("Initializing topic-model at runtime...")
            if self.clean_meth == "vect":
                print("Initializing topic model with stop words vectorizer...")
                vectorizer_model = CountVectorizer(stop_words="english")
                topic_model = BERTopic(vectorizer_model=vectorizer_model, embedding_model=self.bert_model)

            else:
                topic_model = BERTopic(embedding_model=self.bert_model)

            topics, probs = topic_model.fit_transform(data)
            topic_model.save(os.path.join(self.models_path, self.model_file_name))
            return topics, probs, topic_model
