import os

import config
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess

class PreProcess:

    def __init__(self, in_folder, out_folder, project):
        """
        PreProcess takes as parameters input folder, output folder, and project and stores
        sub-folders for all data: text-bodies, split text-bodies, embeddings, and models.
        From PreProcess the TextPreProcess and EmbeddingsPreprocess classes are initialized and called.

        Parameters:
            in_folder (str): Directory path to the folder containing input text data.
            out_folder (str): Directory path to the folder designated for output data.
            project (str): Identifier for the project, used in naming subdirectories.

        Attributes:
            text_bodies_path (str): Directory path where text bodies are stored.
            emb_path (str): Directory path where embeddings are stored.
            model_path (str): Directory path where models are stored.
            split_texts_path (str): Directory path where split texts are stored as .pkl dict file.
        """

        self.in_folder = in_folder
        self.project = project
        self.text_bodies_path = os.path.join(in_folder, project, "text_bodies")

        self.out_folder = out_folder
        self.split_texts_path = os.path.join(out_folder, project, "texts")
        self.emb_path = os.path.join(out_folder, project, "embeddings")
        self.model_path = os.path.join(out_folder, project, "models")

    def initialize_texts(self, splits_from_file, text_clean_method, text_split_size):
        """
        Initialize texts

        Args:
            splits_from_file: Load split texts saved in file or read and split text bodies at runtime.
            text_clean_method: Text cleaning method.
            text_split_size: Text split size.

        Returns:
            texts (list[str]): split texts
        """
        texts_pp = TextPreProcess(splits_from_file, self.text_bodies_path, self.split_texts_path, text_clean_method, text_split_size)
        texts = texts_pp.get_texts()
        return texts

    def initialize_embeddings(self, emb_from_file, data, text_clean_method, text_split_size):
        """
        Initialize embeddings

        Args:
            emb_from_file: Load embeddings saved in file or initialize embeddings at runtime based on text data.
            data: Text data used for generating embeddings.
            text_clean_method: Text cleaning method.
            text_split_size: Text split size.

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector.
        """
        embeddings_pp = EmbeddingsPreProcess(emb_from_file, self.emb_path, text_clean_method, text_split_size)
        embeddings = embeddings_pp.get_embeddings(data)
        return embeddings
