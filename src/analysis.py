from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import time
import os
import pickle

import config

class Analysis:

    def __init__(self, out_path, model_from_file, mod_emb_from_file):
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

        self.models_path = os.path.join(out_path, "models")
        self.emb_path = os.path.join(out_path, "embeddings")

        self.mod_emb_from_file = mod_emb_from_file
        self.model_from_file = model_from_file

        self.clean_meth = config.parameters["clean_meth"]
        self.split_size = config.parameters["split_size"]
        self.chunk_size = config.parameters["chunk_size"]
        self.bert_model = config.parameters["bert_model"]

        if self.split_size == "chunk":
            self.model_file_name = f"bertopic_model_{self.bert_model}_{self.split_size}{self.chunk_size}_{self.clean_meth}"
            self.emb_name = f"embeddings_{self.bert_model}_{self.split_size}{self.chunk_size}_{self.clean_meth}.pkl"
        else:
            self.model_file_name = f"bertopic_model_{self.bert_model}_{self.split_size}_{self.clean_meth}"
            self.emb_name = f"embeddings_{self.bert_model}_{self.split_size}_{self.clean_meth}.pkl"

        self.merged = False

    def initialize_topic_model(self, data):
        if self.model_from_file:
            return self.load_topic_model()
        else:
            return self.generate_topic_model(data)

    def load_topic_model(self):
        print(f"Topic-model name: {self.model_file_name} \nReading topic-model from file...")
        start = time.time()
        topic_model = BERTopic.load(os.path.join(self.models_path, self.model_file_name))
        print(f"Time elapsed to load topic-model: {time.time() - start:.3f} seconds\n")
        return topic_model

    def generate_topic_model(self, data):
        print("Initializing topic-model at runtime...")

        # Applying Vectorization to clean text
        if self.clean_meth == "vect":
            print("Initializing topic model with stop words vectorizer...")
            vectorizer_model = CountVectorizer(stop_words="english")
            topic_model = BERTopic(vectorizer_model=vectorizer_model, embedding_model=self.bert_model)
        else:
            topic_model = BERTopic(embedding_model=self.bert_model)

        # Initializing topic model with pre-trained text embeddings or from text data
        if self.mod_emb_from_file:
            print("Generating topic-model with pre-trained embeddings...")
            with open(os.path.join(self.emb_path, self.emb_name), "rb") as file:
                data_dict = pickle.load(file)
                embeddings = data_dict['embeddings']
            topics, probs = topic_model.fit_transform(data, embeddings)
        else:
            print("Generating topic-model from text data...")
            topics, probs = topic_model.fit_transform(data)

        # Saving topic-model
        topic_model.save(os.path.join(self.models_path, self.model_file_name))
        return topic_model

    def get_model_file_name(self):
        return self.model_file_name

