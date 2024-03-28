import os

class InitFolders:

    def __init__(self, project_root, project, year="2022"):
        """
        InitFolders takes as parameters project_root, project, and year and creates folder structure
        initializes and saves
        sub-folders for all data: text-bodies, split text-bodies, embeddings, and models, and labels.

        Parameters:
            project_root (str): Path of project root, should be path to NLP-Pipeline-BERT.
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

        self.text_bodies_path = os.path.join(self.input_folder, year, "text_bodies")
        self.output_folder = os.path.join(self.project_root, "output", project, year)

        self.split_texts_path = os.path.join(self.output_folder, "texts")
        self.emb_path = os.path.join(self.output_folder, "embeddings")
        self.model_path = os.path.join(self.output_folder, "models")

        self.create_output_folders()

    def create_output_folders(self):
        """
        Create output folders if they do not exist
        """
        for folder in [self.output_folder, self.split_texts_path, self.emb_path, self.model_path]:
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
