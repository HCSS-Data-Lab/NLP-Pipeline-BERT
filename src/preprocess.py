import os

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

    def initialize_data(self):
        """
        TODO:
        Implement this to handle reading embeddings from file or not.
        If emb_from_file, it is not required to do all input text initialize
        # Also, split up embeddings.pkl into text.pkl and embeddings.pkl (now the .pkl contains both)
        """
        pass

    def initialize_texts(self, text_clean_method="def", text_split_size="chunk"):
        """
        Initialize texts from folder text_bodies_path

        Args:
            text_clean_method (str): def, vect, or ft
            text_split_size (str): body, chunk, sentence, or sentence-pairs

        Returns:
            data (list[str]): texts
        """
        if os.listdir(self.text_bodies_path):
            texts_pp = TextPreProcess(self.text_bodies_path, clean_meth=text_clean_method, split_size=text_split_size)
            data = texts_pp.get_texts(split_size=text_split_size)
            return data
        else:
            raise ValueError("Folder input/project/text_bodies does not contain any files")

    def initialize_embeddings(self, data, emb_from_file, text_clean_method="def", text_split_size="chunk"):
        """
        Initialize embeddings by reading from file or initializing embeddings at runtime
        from the input data and using a BERT-model specified in config.

        Args:
            data (list[str]): text data
            emb_from_file (bool): read embeddings from file or not
            text_clean_method (str): text clean method
            text_split_size (str): text split size

        Returns:
            data (list[str]): text data
            embeddings (): 768-dim text embeddings
        """
        embeddings_pp = EmbeddingsPreProcess(self.emb_path, emb_from_file, text_clean_method, text_split_size)
        data, embeddings = embeddings_pp.get_embeddings(data)
        return data, embeddings

    def get_project(self):
        return self.project

    def get_text_bodies_path(self):
        return self.text_bodies_path
