from bertopic import BERTopic
import umap
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import PartOfSpeech
from sklearn.feature_extraction.text import CountVectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer

import time
import os
import pickle

import config

def bool_ind(bool_val):
    return "T" if bool_val else "F"

class Analysis:

    def __init__(self, out_path):
        """
        Class Analysis handles running the topic modeling analysis with the BERTopic module.

        Parameters:
            out_path (str): path to output folder with folders "models" and "embeddings"

        Attributes:
            model_from_file (bool): Use model saved in file or not
            mod_emb_from_file (bool): Use embeddings from file for model or not
            clean_meth (str): Text clean method. (def for default, ft for filter-text function,
                              vect for vectorization param in BERTopic)
            split_size (str): Text split size. (chunk, sentence, or sentence-pairs)
            chunk_size (str): size or number of characters in text chunks
            bert_model (str): Pre-trained sentence BERT model name, defined in config.
            use_mmr (bool): Boolean indicator whether to use MMR for topic fine-tuning or not.
            model_file_name (str): name for topic-model when saving/loading from file.
            emb_name (str): name for embeddings used for saving/loading from file.
        """

        self.project_name = os.path.basename(out_path)

        self.models_path = os.path.join(out_path, "models")
        self.emb_path = os.path.join(out_path, "embeddings")

        self.model_from_file = config.LOAD_TOPIC_MODEL_FROM_FILE
        self.mod_emb_from_file = config.LOAD_MODEL_EMBEDDINGS_FROM_FILE

        self.clean_meth = config.texts_parameters["clean_meth"]
        self.split_size = config.texts_parameters["split_size"]
        self.chunk_size = config.texts_parameters["chunk_size"]

        self.bert_model = config.model_parameters["bert_model"]
        self.bert_model_str = self.bert_model.split("/")[-1]  # When model name is like 'mixedbread-ai/mxb...', only take the second part

        self.use_mmr = config.bertopic_parameters["use_mmr"]
        self.use_pos = config.bertopic_parameters["use_pos"]
        self.update_topics = config.bertopic_parameters["update_topics"]
        self.use_keyphrase = config.bertopic_parameters["use_keyphrase"]

        if self.split_size == "chunk":
            self.model_file_name = f"bertopic_model_{self.bert_model_str}_{self.split_size}{self.chunk_size}_{self.clean_meth}{self.get_repr_str()}"
            self.emb_name = f"embeddings_{self.bert_model_str}_{self.split_size}{self.chunk_size}_{self.clean_meth}.pkl"
        else:
            self.model_file_name = f"bertopic_model_{self.bert_model_str}_{self.split_size}_{self.clean_meth}{self.get_repr_str()}"
            self.emb_name = f"embeddings_{self.bert_model_str}_{self.split_size}_{self.clean_meth}.pkl"

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
            data (list[str]): text data

        Returns:
            topic_model (BERTopic): trained topic model object

        """
        print("Initializing topic-model at runtime...")
        # Get topic-model object
        topic_model = self.get_topic_model_obj()

        # Train topic-model
        topic_model = self.train_topic_model(topic_model, data)

        # Update topic-model
        topic_model = self.update_topic_model(topic_model, data)

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
            print("Initializing topic model with stop words vectorizer.")
            vectorizer_model = CountVectorizer(ngram_range=(1, 3),
                                               # stop_words="english",
                                               stop_words=['de', 'het', 'een', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'of', 'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 'uit', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', 'u', 'want', 'nog', 'zal', 'me', 'zij', 'nu', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hun', 'dus', 'alles', 'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'kunnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'uw', 'iemand', 'geweest', 'andere'],
                                               min_df=0.01,
                                               lowercase=False)

            if self.use_keyphrase:
                print("Using KeyPhrase as CountVectorizer.")
                vectorizer_model = KeyphraseCountVectorizer(stop_words="english",
                                                            spacy_pipeline="en_core_web_sm")

        # Conditionally set representation_model if using MMR for topic fine-tuning
        if self.use_mmr:
            print("Applying topic fine-tuning with MMR.")
            representation_model = MaximalMarginalRelevance(diversity=config.bertopic_parameters["mmr_diversity"])

        if self.use_pos:
            print("Applying topic fine-tuning with Parts Of Speech.")
            # representation_model = PartOfSpeech(config.parameters['spacy_mod_pos'])
            representation_model = PartOfSpeech(config.bertopic_parameters['spacy_mod_pos'], pos_patterns=config.bertopic_parameters["pos_patterns"])

        umap_model = umap.UMAP(n_neighbors=15,
                               n_components=5,
                               min_dist=0.0,
                               metric='cosine',
                               low_memory=False,
                               random_state=config.bertopic_parameters["random_state"])

        # Create the topic model
        topic_model = BERTopic(vectorizer_model=vectorizer_model,
                               embedding_model=self.bert_model,
                               representation_model=representation_model,
                               umap_model=umap_model)
        return topic_model

    def train_topic_model(self, topic_model, data):
        """
        Train topic-model object on data, either pre-trained text embeddings or text data

        Args:
            topic_model (BERTopic): topic-model object, untrained
            data (list[str]): text data

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

    def update_topic_model(self, topic_model, data):
        if self.update_topics:
            print("Updating topic-model...")
            vectorizer_model = CountVectorizer(ngram_range=(1, 3),
                                               stop_words="english",
                                               min_df=0.01)
            topic_model.update_topics(data, vectorizer_model=vectorizer_model)
        return topic_model

    def get_model_file_name(self):
        return self.model_file_name

    def get_repr_str(self):
        """
        Get str of representation model values, used for bertopic model file name;
        T means preceding value is True, F means it is False.
        mmr: use MMR or not;
        p: use PoS (Parts-of-Speech) for topic fine-tuning or not;
        kp: use KeyPhrase as CountVectorizer to filter stop words and noun phrases.

        Returns:
            str: representation model values in str
        """
        return f"_mmr{bool_ind(self.use_mmr)}_p{bool_ind(self.use_pos)}_kp{bool_ind(self.use_keyphrase)}"