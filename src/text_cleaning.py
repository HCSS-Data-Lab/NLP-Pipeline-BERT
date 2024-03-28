"""
Text cleaning .py file

Best if this is part of the preprocessing repository that we will make

"""
import os
import re
from tqdm import tqdm

import config

class TextCleaning:

    def __init__(self, project_root, project, year):
        self.project = project
        self.input_folder = os.path.join(project_root, "input", project)
        self.year = year
        self.raw_texts_path = os.path.join(self.input_folder, year, "raw_texts")

        if not os.path.isdir(self.raw_texts_path):
            raise ValueError(f"Input folder {self.input_folder}/{self.year} should contain a folder 'raw_texts' with texts to clean.")

        self.output_folder = os.path.join(self.input_folder, year, "text_bodies")  # Saving cleaned texts in folder text_bodies
        os.makedirs(self.output_folder, exist_ok=True)

        self.regex = config.clean_parameters[f"regex_{self.project}"]

    def read_clean_raw_texts(self):
        """
        Function to read and clean raw texts, saving them to self.output_folder
        """
        text_names = sorted([text_file for text_file in os.listdir(self.raw_texts_path) if text_file.endswith('.txt')])
        print(f'{f"Number of texts in folder for year {self.year}:":<65}{len(text_names):>10}')
        print("Cleaning texts...")

        for name in tqdm(text_names):
            with open(os.path.join(self.raw_texts_path, name), "r", encoding="utf-8") as file:
                text_body = file.read()
                cleaned_body = re.sub(self.regex, "", text_body)

            out_name = "clean_" + name
            with open(os.path.join(self.output_folder, out_name), 'w+', encoding="utf-8") as out_file:
                out_file.write(cleaned_body)
