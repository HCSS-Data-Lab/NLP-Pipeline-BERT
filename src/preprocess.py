import os
import pickle

import config
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess

class PreProcess:

    def __init__(self, in_folder, out_folder, project):
        """
        PreProcess stores the variables required for preprocessing, so reading input text,
        splitting input text, and initializing embeddings. The class initializes the paths
        for input texts, embeddings and models based on the provided input and output folder
        and project name.

        Parameters:
            in_folder (str): Directory path to the folder containing input text data.
            out_folder (str): Directory path to the folder designated for output data.
            project (str): Identifier for the project, used in naming subdirectories.
            data_from_file (bool): Indicator to read data (split texts and embeddings)
                                from file or initialize them at runtime.

        Attributes:
            in_folder (str):
            out_folder (str):
            project (str):
            text_bodies_path (str): The full path to where text bodies for the project are stored.
            emb_path (str): The full path to where embeddings for the project should be saved.
            model_path (str): The full path to where models for the project should be saved.
        """

        self.in_folder = in_folder
        self.project = project
        self.text_bodies_path = os.path.join(in_folder, project, "text_bodies")

        self.out_folder = out_folder
        self.emb_path = os.path.join(out_folder, project, "embeddings")
        self.model_path = os.path.join(out_folder, project, "models")
        self.split_texts_path = os.path.join(out_folder, project, "texts")

        self.def_clean_meth = config.texts_pp_params["def_clean_meth"]  # Default text clean method from config
        self.def_split_size = config.texts_pp_params["def_split_size"]  # Default text split size from config

    def initialize_texts(self, split_texts_from_file, text_clean_method, text_split_size):
        texts_pp = TextPreProcess(split_texts_from_file, self.text_bodies_path, self.split_texts_path, text_clean_method, text_split_size)
        texts = texts_pp.get_texts()
        return texts

    def initialize_embeddings(self, data, text_clean_method, text_split_size):
        # if self.emb_from_file:
        #     return self.initialize_emb_from_file(data, text_clean_method, text_split_size)
        # else:
        #     return self.initialize_emb_runtime(data, text_clean_method, text_split_size)
        pass

    # def initialize_emb_runtime(self, data, clean_method, split_size):
    #     """
    #     Initialize embeddings at runtime from the input data and using a
    #     BERT-model specified in config.
    #
    #     Args:
    #         data (list[str]): text data
    #         clean_method (str): text clean method
    #         split_size (str): text split size
    #
    #     Returns:
    #         data (list[str]): text data
    #         embeddings (): 768-dim text embeddings
    #     """
    #     embeddings_pp = EmbeddingsPreProcess(self.emb_path, clean_method, split_size)
    #     embeddings = embeddings_pp.generate_embeddings_at_runtime(data)
    #     return embeddings
    #
    # def get_project(self):
    #     return self.project

    def get_text_bodies_path(self):
        return self.text_bodies_path
