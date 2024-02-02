import os

import config
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess

class PreProcess:

    def __init__(self, in_folder, out_folder, project, data_from_file):
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

        self.data_from_file = data_from_file

    def initialize_data(self,
                        text_clean_method=config.texts_pp_params["def_clean_meth"],
                        text_split_size=config.texts_pp_params["def_split_size"]):
        """
        TODO:
        Implement this to handle reading embeddings from file or not.
        If emb_from_file, it is not required to do all input text initialize
        # Also, split up embeddings.pkl into text.pkl and embeddings.pkl (now the .pkl contains both)
        """

        if self.data_from_file:
            # texts from file
            # embeddings from file
        else:
            data = self.initialize_texts_runtime(text_clean_method, text_split_size)
            embeddings = self.initialize_emb_runtime(data, text_clean_method, text_split_size)
            return data, embeddings

    def initialize_texts_from_file(self):
        """ Read splits texts from .pkl """
        pass

    def initialize_emb_from_file(self):
        """ Read embeddings from .pkl """

    def initialize_texts_runtime(self, clean_method, split_size):
        """
        Initialize (read and split) texts from folder self.text_bodies_path at runtime

        Args:
            clean_method (str): def, vect, or ft
            split_size (str): body, chunk, sentence, or sentence-pairs

        Returns:
            data (list[str]): texts
        """
        if os.listdir(self.text_bodies_path):
            texts_pp = TextPreProcess(self.text_bodies_path, clean_meth=clean_method)
            input_texts = texts_pp.read_input_texts()
            split_texts = texts_pp.split_texts(input_texts, split_size)
            return split_texts
        else:
            raise ValueError("Folder input/project/text_bodies does not contain any files")

    def initialize_emb_runtime(self, data, clean_method, split_size):
        """
        Initialize embeddings at runtime from the input data and using a
        BERT-model specified in config.

        Args:
            data (list[str]): text data
            emb_from_file (bool): read embeddings from file or not
            text_clean_method (str): text clean method
            text_split_size (str): text split size

        Returns:
            data (list[str]): text data
            embeddings (): 768-dim text embeddings
        """
        embeddings_pp = EmbeddingsPreProcess(self.emb_path, clean_method, split_size)
        embeddings = embeddings_pp.generate_embeddings_at_runtime(data)

        return embeddings


    def get_project(self):
        return self.project

    def get_text_bodies_path(self):
        return self.text_bodies_path
