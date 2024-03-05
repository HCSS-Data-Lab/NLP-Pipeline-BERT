import os
import config
from src.texts_pp import TextPreProcess
from src.embeddings_pp import EmbeddingsPreProcess
from src.red_embeddings_pp import RedEmbeddingsPreProcess

class PreProcess:

    def __init__(self, project_root, project):
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
        if self.project == "ParlaMint":
            self.text_bodies_path = os.path.join(self.project_root, 'input', project, "2014")
        else:
            self.text_bodies_path = os.path.join(self.project_root, 'input', project, "text_bodies")

        self.split_texts_path = os.path.join(self.project_root, 'output', project, "texts")
        self.emb_path = os.path.join(self.project_root, 'output', project, "embeddings")
        self.model_path = os.path.join(self.project_root, 'output', project, "models")

        self.create_output_folders()

    def initialize_texts(self, splits_from_file):
        """
        Initialize texts

        Args:
            splits_from_file: Load split texts saved in file or read and split text bodies at runtime.

        Returns:
            texts (list[str]): split texts
        """
        texts_pp = TextPreProcess(splits_from_file, self.text_bodies_path, self.split_texts_path)
        texts = texts_pp.get_texts()
        return texts

    def initialize_embeddings(self, emb_from_file, data):
        """
        Initialize embeddings

        Args:
            emb_from_file: Load embeddings saved in file or initialize embeddings at runtime based on text data.
            data: Text data used for generating embeddings.

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        embeddings_pp = EmbeddingsPreProcess(emb_from_file, self.emb_path)
        embeddings = embeddings_pp.get_embeddings(data)
        return embeddings

    def initialize_red_embeddings(self, red_from_file, embeddings):
        """
        Initialize reduced embeddings, ie embeddings mapped to 2 dimensions

        Args:
            red_from_file (bool): Boolean indicator whether to load reduced embeddings saved in file or initialize at runtime
            embeddings (torch.Tensor): text embeddings

        Returns:
            reduced_embeddings (np.ndarray): reduced embeddings as 2-dim np array
        """
        red_emb_pp = RedEmbeddingsPreProcess(red_from_file, self.emb_path)
        reduced_embeddings = red_emb_pp.get_red_embeddings(embeddings)
        return reduced_embeddings

    def create_output_folders(self):
        """
        Create output folders if they do not exist
        """
        folders = [os.path.join(self.project_root, 'output'), self.split_texts_path, self.emb_path, self.model_path]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
