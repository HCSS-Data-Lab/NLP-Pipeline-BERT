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
    print(f'{"Number of shortened texts:":<65}{len(chunks_out):>10}')
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


class TextPreProcess:

    def __init__(self, text_bodies_path, split_texts_path, clean_meth=config.texts_pp_params["def_clean_meth"], split_size=config.texts_pp_params["def_split_size"]):
        """
        Class TextPreProcess stores the path where input texts are stored, how they should
        be cleaned and in what size they are split up.

        Parameters:
            text_bodies_path (str): path where text bodies are stored
            clean_meth (str, optional): text cleaning method

        Attributes:
            path (str):
            clean_meth (str):
            text_bodies (list[str]): list with text bodies
        """

        self.path = text_bodies_path
        self.split_texts_path = split_texts_path
        self.clean_meth = clean_meth  # Text clean method
        self.split_size = split_size
        self.text_bodies = []  # Initialize text_bodies as []

    def load_texts_from_file(self):
        print(f"Reading split text elements from file...")
        with open(os.path.join(self.split_texts_path, f"texts_{self.split_size}_{self.clean_meth}.pkl"),
                  "rb") as file:
            data_dict = pickle.load(file)
        return data_dict['texts']

    def read_split_texts_runtime(self):
        pass

    def read_input_texts(self):
        """
        Read text bodies saved as .txt files in text_bodies_path,
        saving result to attribute self.text_bodies.
        """
        text_names = sorted([text_file for text_file in os.listdir(self.path) if text_file.endswith('.txt')])
        print(f'{"Number of texts in folder:":<65}{len(text_names):>10}')

        texts = []
        for text in text_names:
            with open(os.path.join(self.path, text), "r", encoding="utf-8") as file:
                text_body = file.read()
            texts.append(text_body)
        return texts

    def split_filter_texts(self, texts):
        """
        Split texts and filter result if clean_meth is "ft"

        Returns:
            texts (list[str]): split and filtered texts

        """
        texts = self.split_texts(texts)

        if self.clean_meth == "ft":
            texts = filter_texts(texts)

        return texts

    def split_texts(self, texts, split_size=config.texts_pp_params["def_split_size"]):
        """
        Split text bodies into given split size

        Args:
            texts (list[str]): text bodies
            split_size (str): body, chunk, sentence or sentence-pairs

        Returns:
            list[str]: split texts
        """
        if split_size == "chunk":
            return chunk_texts(texts)

        elif split_size == "sentence":
            return sentencize_text(texts)

        elif split_size == "sentence-pairs":
            sentences = sentencize_text(texts)
            return pair_sentences(sentences)

        else:
            raise ValueError(
                f"split_size: {split_size} is undefined. Valid options are 'chunk', 'sentence', or 'sentence-pairs'.")

    def get_text_bodies(self):
        return self.text_bodies

    def get_text_bodies_path(self):
        return self.path

