import os
from tqdm import tqdm
import spacy
import pickle

import config

def make_chunks(text, max_length):
    """
    Chunk individual text to pieces of max length

    Args:
        text (str):
        max_length (int):

    Returns:
        chunks (list[str]): text in chunks
    """
    chunks = []
    while text:
        chunk = text[:max_length]
        chunks.append(chunk)
        text = text[max_length:]
    return chunks

def pair_sentences(sentences):
    """
    Pair sentences together

    Args:
        sentences (list[str]):

    Returns:
        list[str]: paired sentences
    """
    return [(sentences[i]+""+sentences[i+1] if i+1 < len(sentences) else sentences[i]) for i in range(0, len(sentences), 2)]

def chunk_texts(texts, chunk_size=1000):
    """
    Chunks texts into segments of specified chunk size.

    Args:
        texts (list[str]): input texts
        chunk_size (int): The length of each chunked text segment.

    Returns:
        list[str]: A list containing the chunked text segments.
    """
    chunks_out = []
    for text in texts:
        chunks = make_chunks(text, chunk_size)
        chunks_out.extend(chunks)
    return chunks_out


def sentencize_text(texts):
    """
    TODO:
    - Use Spacy Sentencizer to make sentences, splitting on . is imprecise

    Splits input texts into sentences.

    Args:
        texts (list[str]): text bodies

    Returns:
        list[str]: A list of sentences extracted from the input texts.
    """
    return [sentence for t in texts for sentence in t.split(".")]

def filter_texts(texts):
    """
    Filter text on stopwords and punctuation using Spacy, save lemmatized words

    Args:
        texts (list[str]): input texts

    Returns:
        lemmas (list[str]): filtered text
    """
    print("Filtering texts...")
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    lemmas = []
    for doc in tqdm(nlp.pipe(texts, batch_size=30), total=len(texts)):
        filtered_lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Filter stop words, punctuation and lemmatize
        lemmas.append(" ".join(filtered_lemmas))
    return lemmas


class TextPreProcess:

    def __init__(self, split_texts_from_file, text_bodies_path, split_texts_path, clean_meth, split_size):
        """
        Class TextPreProcess stores the path where input texts are stored, how they should
        be cleaned and in what size they are split up.

        Parameters:
            text_from_file (bool): read split texts from file or read and split and runtime
            text_bodies_path (str): path where text bodies are stored
            clean_meth (str, optional): text cleaning method

        Attributes:
            path (str):
            clean_meth (str):
            text_bodies (list[str]): list with text bodies
        """

        self.split_texts_from_file = split_texts_from_file
        self.bodies_path = text_bodies_path
        self.split_texts_path = split_texts_path
        self.clean_meth = clean_meth  # Text clean method
        self.split_size = split_size  # Text split size
        self.text_bodies = []  # Initialize text_bodies as []

    def get_texts(self):
        if self.split_texts_from_file:
            return self.load_split_texts()
        else:
            return self.generate_split_texts()

    def load_split_texts(self):
        print(f"Reading split text elements from file...")
        path = os.path.join(self.split_texts_path, f"texts_{self.split_size}_{self.clean_meth}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as file:
                data_dict = pickle.load(file)
            return data_dict['texts']
        else:
            raise ValueError(
                f"Folder output/project/texts does not contain text .pkl dictionary file with split text size: {self.split_size} and text clean method: {self.clean_meth}. "
                f"Generate it at runtime.")

    def generate_split_texts(self):
        print("Read text bodies and splitting at runtime...")
        if os.listdir(self.bodies_path):  # Empty folder evaluates to 0
            text_bodies = self.read_input_texts()
            split_texts = self.split_texts(text_bodies)

            with open(os.path.join(self.split_texts_path, f"texts_{self.split_size}_{self.clean_meth}.pkl"), "wb") as file:
                pickle.dump({'texts': split_texts}, file, protocol=pickle.HIGHEST_PROTOCOL)

            return split_texts
        else:
            raise ValueError("Folder input/project/text_bodies does not contain any files")

    def read_input_texts(self):
        """
        Read text bodies saved as .txt files in text_bodies_path,
        """
        text_names = sorted([text_file for text_file in os.listdir(self.bodies_path) if text_file.endswith('.txt')])
        print(f'{"Number of texts in folder:":<65}{len(text_names):>10}')

        texts = []
        for text in text_names:
            with open(os.path.join(self.bodies_path, text), "r", encoding="utf-8") as file:
                text_body = file.read()
            texts.append(text_body)
        return texts

    def split_texts(self, texts):
        """
        Split text bodies into split size and apply filter function if clean_meth = "ft" (filter_texts)
        """
        if self.split_size == "chunk":
            splits = chunk_texts(texts)
        elif self.split_size == "sentence":
            splits = sentencize_text(texts)
        elif self.split_size == "sentence-pairs":
            sentences = sentencize_text(texts)
            splits = pair_sentences(sentences)
        else:
            raise ValueError(
                f"split_size: {self.split_size} is undefined. Valid options are 'chunk', 'sentence', or 'sentence-pairs'.")

        print(f"Split size: {self.split_size}")
        print(f"{'Number of split texts:':<65}{len(splits):>10}")
        if self.clean_meth == "ft":
            return filter_texts(splits)
        else:
            return splits

    def get_text_bodies(self):
        return self.text_bodies

    def get_text_bodies_path(self):
        return self.bodies_path

