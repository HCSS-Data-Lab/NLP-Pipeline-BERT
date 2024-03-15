import os

from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess
from src.red_embeddings_pp import RedEmbeddingsPreProcess

class PreProcess:

    def __init__(self, project_root, project, year="2022"):
        """
        PreProcess takes as parameters input folder, output folder, and project and stores
        sub-folders for all data: text-bodies, split text-bodies, embeddings, and models.
        From PreProcess the TextPreProcess and EmbeddingsPreprocess classes are initialized and called.

        Parameters:
            project_root (str): Directory path to the folder containing input text data.
            project (str): Identifier for the project, used in naming subdirectories.

        Attributes:
            text_bodies_path (str): Directory path where text bodies are stored.
            emb_path (str): Directory path where embeddings are stored.
            model_path (str): Directory path where models are stored.
            split_texts_path (str): Directory path where split texts are stored as .pkl dict file.
        """
        self.project_root = project_root
        self.project = project

        self.input_folder = os.path.join(self.project_root, "input", project)
        if not os.path.exists(self.input_folder):
            raise ValueError(f"No project folder in input for project name {self.project}. The path {self.input_folder} does not exist. Create it and make sure it contains a folder text_bodies.")

        self.text_bodies_path = os.path.join(self.input_folder, "text_bodies")
        # if not os.path.exists(self.text_bodies_path):
        #     raise ValueError(f"No text bodies folder. The path {self.text_bodies_path} does not exist. Create it and make sure it contains text bodies as .txt files.")

        if self.project == "ParlaMint":
            self.text_bodies_path = os.path.join(self.text_bodies_path, year)

        self.output_folder = os.path.join(self.project_root, "output", project, year)

        self.split_texts_path = os.path.join(self.output_folder, "texts")
        self.emb_path = os.path.join(self.output_folder, "embeddings")
        self.model_path = os.path.join(self.output_folder, "models")
        self.labels_path = os.path.join(self.output_folder, "labels")

        self.create_output_folders()

    def initialize_texts(self):
        """
        Initialize texts

        Returns:
            texts (list[str]): split texts
        """
        texts_pp = TextPreProcess(self.text_bodies_path, self.split_texts_path)
        texts = texts_pp.get_texts()
        return texts

    def initialize_embeddings(self, data):
        """
        Initialize embeddings

        Args:
            data: Text data used for generating embeddings.

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        embeddings_pp = EmbeddingsPreProcess(self.emb_path)
        embeddings = embeddings_pp.get_embeddings(data)
        return embeddings

    def initialize_red_embeddings(self, embeddings):
        """
        Initialize reduced embeddings, ie embeddings mapped to 2 dimensions

        Args:
            red_from_file (bool): Boolean indicator whether to load reduced embeddings saved in file or initialize at runtime
            embeddings (torch.Tensor): text embeddings

        Returns:
            reduced_embeddings (np.ndarray): reduced embeddings as 2-dim np array
        """
        red_emb_pp = RedEmbeddingsPreProcess(self.emb_path)
        reduced_embeddings = red_emb_pp.get_red_embeddings(embeddings)
        return reduced_embeddings

    def create_output_folders(self):
        """
        Create output folders if they do not exist
        """
        folders = [self.output_folder, self.split_texts_path, self.emb_path, self.model_path, self.labels_path]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def get_input_folder(self):
        return self.input_folder

    def get_text_bodies_path(self):
        return self.text_bodies_path
