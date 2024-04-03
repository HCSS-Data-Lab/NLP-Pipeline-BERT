import os

class DynamicTopicModeling:

    def __init__(self, project_root, project, dtm_years=['2019']):
        self.project_root = project_root
        self.project = project
        self.input_folder = os.path.join(self.project_root, self.project)
        self.dtm_years = dtm_years

    def run_dtm(self):
        for year in self.dtm_years:
            folder_path = os.path.join(self.input_folder, year)
            texts = sorted([text_file for text_file in os.listdir(folder_path) if text_file.endswith('.txt')])



