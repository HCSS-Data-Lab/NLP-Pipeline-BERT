import os

class InitFolders:

    def __init__(self, project_root, dataset_name, year_str, country=None):
        """
        InitFolders takes as parameters project_root, project, and year and creates folder structure
        initializes and saves
        sub-folders for all data: text-bodies, split text-bodies, embeddings, and models, and labels.

        Parameters:
            project_root (str): Path of project root, should be path to NLP-Pipeline-BERT.
            dataset_name (str): Identifier for the project, used in naming subdirectories.
            year_str (str): year string

        Attributes:
            text_bodies_path (str): Directory path where text bodies are stored.
            emb_path (str): Directory path where embeddings are stored.
            model_path (str): Directory path where models are stored.
            split_texts_path (str): Directory path where split texts are stored as .pkl dict file.
            fig_path (str): Directory path where figures are stored.
            RAG_path (str): Dir where RAG data is stored.
            labels_path (str): Dir where labels from RAG are stored.
        """
        self.project_root = project_root
        self.dataset_name = dataset_name

        self.input_folder = os.path.join(self.project_root, "input", dataset_name)
        if not os.path.exists(self.input_folder):
            raise ValueError(f"No project folder in input for dataset name {self.dataset_name}. The path {self.input_folder} does not exist. Create it and make sure it contains a folder text_bodies.")

        self.text_bodies_path = os.path.join(self.input_folder, year_str, "text_bodies")
        if country is not None:
            self.output_folder = os.path.join(self.project_root, "output", dataset_name, country, year_str)
        else:
            self.output_folder = os.path.join(self.project_root, "output", dataset_name, year_str)

        self.split_texts_path = os.path.join(self.output_folder, "texts")
        self.emb_path = os.path.join(self.output_folder, "embeddings")
        self.model_path = os.path.join(self.output_folder, "models")
        self.fig_path = os.path.join(self.output_folder, "figures")
        self.RAG_path = os.path.join(self.output_folder, "RAG")
        self.labels_path = os.path.join(self.RAG_path, "labels")

        self.create_output_folders()

    def create_output_folders(self):
        """
        Create output folders if they do not exist
        """
        for folder in [self.output_folder, self.split_texts_path, self.emb_path, self.model_path, self.fig_path, self.RAG_path,  self.labels_path]:
            os.makedirs(folder, exist_ok=True)

    def get_input_folder(self):
        return self.input_folder

    def get_text_bodies_path(self):
        return self.text_bodies_path

    def get_output_folder(self):
        return self.output_folder

    def get_split_texts_path(self):
        return self.split_texts_path

    def get_emb_path(self):
        return self.emb_path

    def get_model_path(self):
        return self.model_path

    def get_fig_path(self):
        return self.fig_path

    def get_rag_path(self):
        return self.RAG_path

    def get_labels_path(self):
        return self.labels_path
