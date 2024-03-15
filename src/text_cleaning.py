"""
Text cleaning .py script

Best if this is part of the preprocessing repository

"""
import os
import re
from tqdm import tqdm

import config

class TextCleaning:

    def __init__(self, project_root, project):
        self.project = project
        self.input_folder = os.path.join(project_root, "input", project)
        self.raw_texts_path = os.path.join(self.input_folder, "raw_texts")

        if not os.path.isdir(self.raw_texts_path):
            raise ValueError(f"Input folder {self.input_folder} should contain a folder 'raw_texts' with texts to clean.")

        self.output_folder = os.path.join(self.input_folder, "text_bodies")  # Saving cleaned texts in folder text_bodies
        if not os.path.exists(self.output_folder):
            print("Creating output folder to save text bodies...")
            os.makedirs(self.output_folder)

    def run_text_cleaning(self):
        regex = config.clean_parameters[f"pattern_{self.project}"]
        self.get_project_texts(regex)

    def get_project_texts(self, regex, save_to_folder=True):
        """
        Get texts for project and saves to folder "Text bodies"

        """
        if self.project == "ParlaMint":
            years = os.listdir(self.raw_texts_path)

            for year in years:
                text_names = sorted([text_file for text_file in os.listdir(os.path.join(self.raw_texts_path, year)) if
                                     text_file.endswith('.txt')])
                print(f'{f"Number of texts in folder for year {year}:":<65}{len(text_names):>10}')

                out_folder = os.path.join(self.output_folder, year)
                os.makedirs(out_folder, exist_ok=True)

                for name in tqdm(text_names):
                    with open(os.path.join(self.raw_texts_path, year, name), "r", encoding="utf-8") as file:
                        text_body = file.read()
                        cleaned_body = re.sub(regex, "", text_body)

                    if save_to_folder:
                        out_name = "clean_" + name
                        with open(os.path.join(out_folder, out_name), 'w+', encoding="utf-8") as out_file:
                            out_file.write(cleaned_body)

        else:
            # Do the same but without year
            pass






