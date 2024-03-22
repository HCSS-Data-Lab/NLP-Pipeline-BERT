import os
import numpy as np
import umap.umap_ as umap

import config

class RedEmbeddingsPreProcess:

    def __init__(self, emb_path):
        """
        Class RedEmbeddingsPreProcess stores the variables used for reduced embeddings
        preprocessing and handles all the functionality of loading or generating
        reduced embeddings.

        Parameters:
            red_from_file (bool): Boolean indicator whether to read reduced embeddings from file or not
            emb_path (str): Directory path where to load/save embeddings.

        Attributes:
            clean_meth (str): Text clean method. (def for default, ft for filter-text function,
                              vect for vectorization param in BERTopic)
            split_size (str): Text split size. (chunk, sentence, or sentence-pairs)
            chunk_size (str): size or number of characters in text chunks
            bert_model (str): Pre-trained sentence BERT model name, defined in config.
            random_state (int): Random state value used in UMAP function, for reproducibility over runs
            red_emb_name (str): reduced embeddings file name, used when saving to file
        """

        self.red_from_file = config.LOAD_REDUCED_EMBEDDINGS_FROM_FILE
        self.path = emb_path
        self.clean_meth = config.texts_parameters["clean_meth"]
        self.split_size = config.texts_parameters["split_size"]
        self.chunk_size = config.texts_parameters["chunk_size"]
        self.bert_model = config.model_parameters["bert_model"]

        self.random_state = config.bertopic_parameters["random_state"]

        if self.split_size == "chunk":
            self.red_emb_name = f"red_embeddings_{self.bert_model}_{self.split_size}{self.chunk_size}_{self.clean_meth}.npy"
        else:
            self.red_emb_name = f"red_embeddings_{self.bert_model}_{self.split_size}_{self.clean_meth}.npy"

    def get_red_embeddings(self, embeddings):
        if self.red_from_file:
            red_embeddings = self.load_red_embeddings()
        else:
            red_embeddings = self.generate_red_embeddings(embeddings)
        print(f'{"Reduced Embeddings shape:":<65}{str(red_embeddings.shape):>10}\n')
        return red_embeddings

    def load_red_embeddings(self):
        print(f"Reduced embedding file name: {self.red_emb_name}. \nReading reduced embeddings from file...")
        return np.load(os.path.join(self.path, self.red_emb_name))

    def generate_red_embeddings(self, embeddings):
        print("Initializing reduced embeddings at runtime...")
        reduced_embeddings = umap.UMAP(n_neighbors=10,
                                       n_components=2,
                                       min_dist=0.0,
                                       metric='cosine',
                                       random_state=self.random_state).fit_transform(embeddings)

        # Saving reduced embeddings to file
        np.save(os.path.join(self.path, self.red_emb_name), reduced_embeddings)
        return reduced_embeddings
