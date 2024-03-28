import os
from easynmt import EasyNMT
import re
from tqdm import tqdm
import glob

import config


def sentencize_text(text_body):
    """
    Splits input text body into sentences

    Args:
        text_body: str = input text body

    Returns:
        List[str] = sentences

    """
    delimiters = ['.', '?', '!']
    pattern = '\s|'.join(map(re.escape, delimiters))  # Add whitespace \s, regex OR delim |, and add backslash with escape
    return re.split(pattern, text_body)

def latest_file_index(folder, text_names):
    """
    Find latest added file in folder and return its name, return 0 if folder contains no files
    """
    files = glob.glob(os.path.join(folder, "*"))
    if files:
        latest_file_path = max(files, key=os.path.getctime)
        name_prefix = os.path.basename(latest_file_path)  # Name including lang prefix, like en_
        name = remove_lang_prefix(name_prefix)  # Original name
        return text_names.index(name)
    else:
        return 0

def remove_lang_prefix(file_name):
    """
    Remove language prefix of output files, which is given by [lang prefix]_[filename]
    """
    return file_name.split("_")[1:][0]

class Translate:

    def __init__(self, project_root, project, year, model_name="opus-mt"):
        """
        Translate is the class to handle text translation. It takes as input the project root,
        project name, year, target language and translation model name. It initializes the input
        and output folder, and initializes the Neural Machine Translation (NMT) model used for translation

        Parameters:
        - project_root (str): The root directory of the project where the input and output folders will be located.
        - project (str): The name of the specific project within the root directory.
        - year (str): The year associated with the project's data.
        - target_lang (str, optional): The language code (e.g., "en" for English) to which the text will be translated. Defaults to "en".
        - model_name (str, optional): The name of the pre-trained NMT model to be used for translation. Defaults to "opus-mt".

        Attributes:
        - input_folder (str): The path to the folder containing the text bodies to be translated. It is constructed based on the project root, project name, and year.
        - output_folder (str): The path to the folder where the translated text bodies will be stored.
        - target_lang (str): Stores the target language code specified during the class initialization.
        - model (EasyNMT): An instance of the EasyNMT model specified by model_name.

        """
        self.target_lang = config.translate_param["target_lang"]
        self.source_lang = config.translate_param["source_lang"]

        self.input_folder = os.path.join(project_root, "input", project, year, "text_bodies")  # Folder with text bodies
        self.output_folder = os.path.join(project_root, "input", project, year+f"_{self.target_lang}", "text_bodies")  # Output folder with translations
        os.makedirs(self.output_folder, exist_ok=True)

        self.model = EasyNMT(model_name)

    def translate_text(self, from_last_added=True):
        print("Translating texts...")
        text_names = sorted([text_file for text_file in os.listdir(self.input_folder) if text_file.endswith('.txt')])

        if from_last_added:  # Start translating from last added file
            latest_index = latest_file_index(self.output_folder, text_names)
            text_names = text_names[latest_index:]

        for name in tqdm(text_names):
            with open(os.path.join(self.input_folder, name), "r", encoding="utf-8") as file:
                text_body = file.read()

            # # Performance is better when giving sentences as input, not full text;
            # # however, all sentences are concatenated with '.', other delimiters are removed. And with full text is faster
            # sentences = sentencize_text(text_body)
            # sentences_trans = self.model.translate(sentences, target_lang=self.target_lang, source_lang="nl")  # Translated sentences
            # translation = ". ".join(sentences_trans)  # Translated text

            translation = self.model.translate(text_body, target_lang=self.target_lang, source_lang=self.source_lang)

            with open(os.path.join(self.output_folder, f"{self.target_lang}_"+name), "w+", encoding="utf-8") as file:
                file.write(translation)













