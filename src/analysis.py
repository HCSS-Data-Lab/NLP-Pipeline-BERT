from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
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
            out_path (str): path to output folder with folders "models" and "embeddings"
            model_from_file (bool): Use model saved in file or not
            mod_emb_from_file (bool): Use embeddings from file for model or not

        Attributes:

        """

        self.models_path = os.path.join(out_path, "models")
        self.emb_path = os.path.join(out_path, "embeddings")

        self.mod_emb_from_file = mod_emb_from_file
        self.model_from_file = model_from_file

        self.clean_meth = config.parameters["clean_meth"]
        self.split_size = config.parameters["split_size"]
        self.chunk_size = config.parameters["chunk_size"]
        self.bert_model = config.parameters["bert_model"]
        self.use_mmr = config.parameters["use_mmr"]

        if self.split_size == "chunk":
            self.model_file_name = f"bertopic_model_{self.bert_model}_{self.split_size}{self.chunk_size}_{self.clean_meth}_mmr{self.use_mmr}"
            self.emb_name = f"embeddings_{self.bert_model}_{self.split_size}{self.chunk_size}_{self.clean_meth}.pkl"
        else:
            self.model_file_name = f"bertopic_model_{self.bert_model}_{self.split_size}_{self.clean_meth}_mmr{self.use_mmr}"
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
        """
        Load and train topic model at runtime based on given input data

        Args:
            data:

        Returns:

        """
        print("Initializing topic-model at runtime...")

        # Get topic-model object
        topic_model = self.get_topic_model_obj()

        # Train topic-model
        topic_model = self.train_topic_model(topic_model, data)

        # Save topic-model
        topic_model.save(os.path.join(self.models_path, self.model_file_name))
        return topic_model

    def get_topic_model_obj(self):
        """
        Get topic-model object of BERTopic module, applying correct cleaning and using MMR

        Returns:
            topic_model (BERTopic): topic-model object

        """
        # Initializing vectorizer to clean text and representation model, both as None if not used
        vectorizer_model = None
        representation_model = None

        # Conditionally set vectorizer_model if using a stop words vectorizer
        if self.clean_meth == "vect":
            print("Initializing topic model with stop words vectorizer...")
            vectorizer_model = CountVectorizer(stop_words="english")

        # Conditionally set representation_model if using MMR for topic fine-tuning
        if self.use_mmr:
            print("Applying topic fine-tuning with MMR.")
            representation_model = MaximalMarginalRelevance(diversity=0.2)

        # Create the topic model with conditionally added components
        topic_model = BERTopic(vectorizer_model=vectorizer_model,
                               embedding_model=self.bert_model,
                               representation_model=representation_model)
        return topic_model

    def train_topic_model(self, topic_model, data):
        """
        Train topic-model object on data, either pre-trained text embeddings or text data

        Args:
            topic_model:
            data:

        Returns:
            topic_model (BERTopic): trained topic_model object of BERTopic module
        """
        if self.mod_emb_from_file:
            print("Generating topic-model with pre-trained embeddings...")
            with open(os.path.join(self.emb_path, self.emb_name), "rb") as file:
                data_dict = pickle.load(file)
                embeddings = data_dict['embeddings']
            topics, probs = topic_model.fit_transform(data, embeddings)
        else:
            print("Generating topic-model from text data...")
            topics, probs = topic_model.fit_transform(data)
        return topic_model

    def get_model_file_name(self):
        return self.model_file_name

