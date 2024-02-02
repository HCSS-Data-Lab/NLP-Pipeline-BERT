import os
import pickle
from sentence_transformers import SentenceTransformer

import config

class EmbeddingsPreProcess:

    def __init__(self, emb_path, emb_from_file, clean_method, split_size):
        """
        Initializes the EmbeddingsPreProcess object with necessary configurations.

        Parameters:
            emb_path (str): Path where embeddings are stored or will be saved.
            emb_from_file (bool): Flag to indicate whether to load embeddings from a file.
            clean_method (str): Method used for text cleaning.
            split_size (str): Size of text chunks for splitting before embedding.

        Attributes:
            emb_path (str):
            emb_from_file (bool):
            clean_method (str):
            split_size (str):
            embedding_name (str): embeddings filename based on split_size and clean_method
        """

        self.path = emb_path
        self.emb_from_file = emb_from_file
        self.clean_method = clean_method
        self.split_size = split_size
        self.embedding_name = f"embeddings_{split_size}_{clean_method}.pkl"

    def get_embeddings(self, data):
        """
        Retrieves or generates embeddings for the provided data. Method either loads embeddings
        or generates embeddings at runtime using a BERT-model specified in config. It also saves
        the generated embeddings.

        Args:
            data (list[str]): The input data for which embeddings are to be generated

        Returns:
            tuple: A tuple containing the data and their corresponding embeddings.
        """
        if self.emb_from_file:
            return self._load_embeddings_from_file()
        else:
            return self._generate_embeddings_at_runtime(data)

    def _load_embeddings_from_file(self):
        """
        Loads embeddings from file.
        """
        print(f"Embedding file name: {self.embedding_name}\nReading embeddings from file...")
        with open(os.path.join(self.path, self.embedding_name), "rb") as file:
            data_dict = pickle.load(file)
        return data_dict['data'], data_dict['embeddings']

    def _generate_embeddings_at_runtime(self, data):
        """
        Generates embeddings at runtime using a pre-defined model.
        """
        print("Initializing embeddings at runtime...")
        model = SentenceTransformer(config.emb_pp_params["bert_model"])
        embeddings = model.encode(data, show_progress_bar=True)

        with open(os.path.join(self.path, self.embedding_name), "wb") as file:
            pickle.dump({'data': data, 'embeddings': embeddings, 'embedding-text-size': self.split_size}, file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return data, embeddings






