from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from bertopic.representation import PartOfSpeech

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from transformers.pipelines import pipeline
from keyphrase_vectorizers import KeyphraseCountVectorizer
import time
import os
import pickle
import umap
import pandas as pd

import config

def bool_ind(bool_val):
    return "T" if bool_val else "F"

class TopicModeling:

    def __init__(self, out_path):
        """
        Class TopicModeling handles running the topic modeling analysis with the BERTopic module.

        Parameters:
            out_path (str): path to output folder with folders "models" and "embeddings"

        Attributes:
            model_from_file (bool): Use model saved in file or not
            mod_emb_from_file (bool): Use embeddings from file for model or not
            clean_meth (str): Text clean method. (def for default, ft for filter-text function,
                              vect for vectorization param in BERTopic)
            split_size (str): Text split size. (chunk, sentence, or sentence-pairs)
            chunk_size (str): size or number of characters in text chunks
            emb_model (str): Pre-trained sentence BERT model name, defined in config.
            use_mmr (bool): Boolean indicator whether to use MMR for topic fine-tuning or not.
            model_file_name (str): name for topic-model when saving/loading from file.
            emb_name (str): name for embeddings used for saving/loading from file.
        """

        self.project_name = os.path.basename(out_path)

        self.models_path = os.path.join(out_path, "models")
        self.emb_path = os.path.join(out_path, "embeddings")

        self.model_from_file = config.LOAD_TOPIC_MODEL_FROM_FILE
        self.mod_emb_from_file = config.LOAD_MODEL_EMBEDDINGS_FROM_FILE

        self.split_size = config.text_splitting_parameters["split_size"]
        self.chunk_size = config.text_splitting_parameters["chunk_size"]

        self.emb_model = config.model_parameters["emb_model"]
        self.bert_model_str = self.emb_model.split("/")[-1]  # When model name is like 'mixedbread-ai/mxb...', only take the second part

        self.clean_meth = config.tm_parameters["clean_meth"]
        self.use_mmr = config.tm_parameters["use_mmr"]
        self.use_pos = config.tm_parameters["use_pos"]
        self.use_keyphrase = config.tm_parameters["use_keyphrase"]
        self.use_custom_stopwords = config.tm_parameters["use_custom_stopwords"]
        self.use_ctfidf = config.tm_parameters["use_ctfidf"]
        self.update_topics = config.tm_parameters["update_topics"]

        self.emb_name = f"embeddings_{self.bert_model_str}.pkl"
        self.model_file_name = f"bertopic_model_{self.split_size}{self.chunk_size}_{self.bert_model_str}{self.get_repr_str()}"

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
            data (List[str]): text data

        Returns:
            topic_model (BERTopic): trained topic model object
        """
        print("Initializing topic-model at runtime...")
        start = time.time()
        # Get topic-model object
        topic_model = self.get_topic_model_obj()

        # Finetune topic-model
        topic_model = self.finetune_topic_model(topic_model, data)

        # Update topic-model
        if self.update_topics:
            topic_model = self.update_topic_model(topic_model, data)

        # Save topic-model
        topic_model.save(os.path.join(self.models_path, self.model_file_name))
        print(f"Time elapsed to generate topic model at runtime: {time.time() - start:.3f} seconds\n")
        return topic_model

    def get_topic_model_obj(self):
        """
        Get topic-model object of BERTopic module, applying correct cleaning and using MMR

        Returns:
            topic_model (BERTopic): topic-model object
        """
        # Initializing vectorizer to clean text, representation model and c-tf-idf model, None if not used
        vectorizer_model = None
        representation_model = None
        ctfidf_model = None

        # Conditionally set vectorizer_model if using a stop words vectorizer
        if self.clean_meth == "vect":
            print("Initializing topic model with stop words vectorizer.")
            if self.use_custom_stopwords:
                print("Using custom stop words.")
                custom_stop_words = config.stop_words_parameters["custom_stopwords"]
                stop_words = ENGLISH_STOP_WORDS.union(custom_stop_words)
            else:
                print("Using default stop words.")
                stop_words = ENGLISH_STOP_WORDS

            vectorizer_model = CountVectorizer(stop_words=list(stop_words),
                                               **config.countvectorizer_parameters)

            if self.use_keyphrase:
                print("Using KeyPhrase as CountVectorizer.")
                vectorizer_model = KeyphraseCountVectorizer(**config.kpcountvectorizer_parameters)

        # Conditionally set representation_model and ctfidf_model
        if self.use_mmr:
            print(f"Applying topic fine-tuning with MMR. diversity: {config.mmr_parameters['diversity']}")
            representation_model = MaximalMarginalRelevance(**config.mmr_parameters)

        if self.use_pos:
            print("Applying topic fine-tuning with Parts Of Speech.")
            representation_model = PartOfSpeech(**config.pos_parameters)

        if self.use_ctfidf:
            print("Applying custom c-TF-IDF model.")
            ctfidf_model = ClassTfidfTransformer(**config.ctfidf_parameters)

        # UMAP model
        umap_model = umap.UMAP(**config.umap_parameters)

        # Embedding model
        if config.model_parameters['non_st_model']:
            embedding_model = pipeline("feature-extraction", model=self.emb_model)
        else:
            embedding_model = self.emb_model

        # Create the topic model
        topic_model = BERTopic(vectorizer_model=vectorizer_model,
                               embedding_model=embedding_model,
                               representation_model=representation_model,
                               umap_model=umap_model,
                               ctfidf_model=ctfidf_model)
        return topic_model

    def finetune_topic_model(self, topic_model, data):
        """
        Fine-tune topic-model object on data, either pre-trained text embeddings or text data

        Args:
            topic_model (BERTopic): topic-model object, untrained
            data (List[str]): text data

        Returns:
            topic_model (BERTopic): trained topic_model object of BERTopic module
        """
        if self.mod_emb_from_file:
            print("Generating topic-model with embeddings from file...")
            with open(os.path.join(self.emb_path, self.emb_name), "rb") as file:
                data_dict = pickle.load(file)
                embeddings = data_dict['embeddings']
            topics, probs = topic_model.fit_transform(data, embeddings)
        else:
            print("Generating topic-model from text data...")
            topics, probs = topic_model.fit_transform(data)
        return topic_model

    def update_topic_model(self, topic_model, data):
        """
        Update the topic representation with (possibly) new parameters. Useful when running
        analysis with parameter change, retraining topic model is not necessary.
        """
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
        Get str of representation model values, used for bertopic model file name.
        T means preceding value is True, F means it is False.
        mmr: use MMR or not;
        p: use PoS (Parts-of-Speech) for topic fine-tuning or not;
        kp: use KeyPhrase as CountVectorizer to filter stop words and noun phrases or not.

        Returns:
            str: representation model values in str
        """
        # return f"_mmr{bool_ind(self.use_mmr)}_p{bool_ind(self.use_pos)}_kp{bool_ind(self.use_keyphrase)}"
        return f"_ctfidf{bool_ind(self.use_ctfidf)}"

    def save_topic_words(self, topic_model, top_n_topics=None):
        """
        Save the top 10 terms of the first top_n_topics topics to a csv file, saving to 'models' folder
        """
        topic_repr = topic_model.topic_representations_

        if top_n_topics:
            topic_repr = dict(sorted(topic_repr.items(), key=lambda item: item[0])[:top_n_topics])

        data_for_df = []
        for topic_id, words_weights in topic_repr.items():
            # Extract only the words, ignoring the weights
            words = ', '.join([word for word, weight in words_weights])
            data_for_df.append({"Topic ID": topic_id, "Words": words})

        # Create a DataFrame
        df = pd.DataFrame(data_for_df)

        # Write the DataFrame to a CSV file
        df.to_excel(os.path.join(self.models_path, f"topic_words_{top_n_topics}.xlsx"), index=False)

    def get_model_str(self):
        return self.bert_model_str
