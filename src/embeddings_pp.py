import os
import pickle
from sentence_transformers import SentenceTransformer

import config

class EmbeddingsPreProcess:

    def __init__(self, emb_from_file, emb_path, clean_meth, split_size):
        """
        Initializes the EmbeddingsPreProcess object with necessary configurations.

        Parameters:
            emb_path (str): Path where embeddings are stored or will be saved.
            clean_method (str): Method used for text cleaning.
            split_size (str): Size of text chunks for splitting before embedding.

        Attributes:
            emb_path (str):
            clean_method (str):
            split_size (str):
            embedding_name (str): embeddings filename based on split_size and clean_method
        """

        self.emb_from_file = emb_from_file
        self.path = emb_path
        self.clean_meth = clean_meth
        self.split_size = split_size
        self.embedding_name = f"embeddings_{split_size}_{clean_meth}.pkl"

    def get_embeddings(self, data):
        if self.emb_from_file:
            embeddings = self.load_embeddings()
        else:
            embeddings = self.generate_embeddings(data)
        print(f'{"Embeddings shape:":<65}{str(embeddings.shape):>10}')
        return embeddings

    def load_embeddings(self):
        """
        Loads embeddings from file.
        """
        path = os.path.join(self.path, self.embedding_name)
        if os.path.exists(path):
            print(f"Embedding file name: {self.embedding_name}. Reading embeddings from file...")
            with open(os.path.join(self.path, self.embedding_name), "rb") as file:
                data_dict = pickle.load(file)
            return data_dict['embeddings']
        else:
            raise ValueError(
                f"Folder output/project/embeddings does not contain emb .pkl dict file with split text size: {self.split_size} and text clean method: {self.clean_meth}. "
                f"Generate it at runtime.")

    def generate_embeddings(self, data):
        """
        Generates embeddings at runtime using a pre-defined model.
        """
        print("Initializing embeddings at runtime...")
        model = SentenceTransformer(config.emb_pp_params["bert_model"])
        embeddings = model.encode(data, show_progress_bar=True)

        with open(os.path.join(self.path, self.embedding_name), "wb") as file:
            pickle.dump({'embeddings': embeddings, 'embedding-text-size': self.split_size}, file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings






