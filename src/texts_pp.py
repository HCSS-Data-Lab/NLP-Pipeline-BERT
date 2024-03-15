import os
from tqdm import tqdm
import spacy
import pickle
import re

import config

def make_chunks(text, max_length):
    """
    Chunk individual text to pieces of max length

    Args:
        text (str)
        max_length (int)

    Returns:
        chunks (list[str]): text in chunks
    """
    chunks = []
    while text:
        chunk = text[:max_length]
        chunks.append(chunk)
        text = text[max_length:]
    return chunks

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

def pair_sentences(sentences):
    """
    Pair sentences together

    Args:
        sentences (list[str])

    Returns:
        list[str]: paired sentences
    """
    return [(sentences[i]+""+sentences[i+1] if i+1 < len(sentences) else sentences[i]) for i in range(0, len(sentences), 2)]

def filter_texts(texts):
    """
    Filter text on stopwords and punctuation using Spacy, save lemmas of words

    Args:
        texts (list[str]): input texts

    Returns:
        lemmas (list[str]): filtered, lemmatized text
    """
    print("Filtering texts...")
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    lemmas = []
    for doc in tqdm(nlp.pipe(texts, batch_size=30), total=len(texts)):
        filtered_lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Filter stop words, punctuation and lemmatize
        lemmas.append(" ".join(filtered_lemmas))
    return lemmas

class TextPreProcess:

    def __init__(self, text_bodies_path, splits_path):
        """
        Class TextPreProcess stores the variables needed in text preprocessing: splits from file,
        text-bodies path, split-texts path, text clean method and text split size. It also handles
        all functionality of text preprocessing: loading split texts from file or reading text bodies
        and splitting texts at runtime.

        Parameters:
            text_bodies_path (str): Directory path where text bodies are stored.
            splits_path (str): Directory path where split texts are stored.

        Attributes:
            splits_from_file (bool): Load split texts from file or read and split text bodies at runtime
            clean_meth (str): Text clean method. (def for default, ft for filter-text function,
                              vect for vectorization param in BERTopic)
            split_size (str): Text split size. (chunk, sentence, or sentence-pairs)
            texts_split_name (str): Text split .pkl dictionary filename
        """

        self.splits_from_file = config.LOAD_TEXT_SPLITS_FROM_FILE
        self.bodies_path = text_bodies_path
        self.splits_path = splits_path

        self.project = self.get_project_name()

        self.clean_meth = config.texts_parameters["clean_meth"]
        self.split_size = config.texts_parameters["split_size"]
        self.chunk_size = config.texts_parameters["chunk_size"]

        if self.split_size == "chunk":
            self.texts_split_name = f"texts_{self.split_size}{self.chunk_size}_{self.clean_meth}.pkl"
        else:
            self.texts_split_name = f"texts_{self.split_size}_{self.clean_meth}.pkl"

    def get_texts(self):
        """
        Get text data calls loads_split_texts() to load split texts from file or
        generate_split_texts() to read text bodies and split them at runtime.

        Returns:
            texts (list[str]): split text data
        """
        if self.splits_from_file:
            texts = self.load_split_texts()
        else:
            texts = self.generate_split_texts()
        print(f"Split size: {self.split_size}")
        print(f'{"Number of split texts:":<65}{len(texts):>10}\n')
        return texts

    def load_split_texts(self):
        """
        Load split texts stored as .pkl dict from self.splits_path

        Returns:
            list[str]: split texts from file

        """
        path = os.path.join(self.splits_path, self.texts_split_name)
        if os.path.exists(path):
            print(f"Split texts file name: {self.texts_split_name}. \nReading split text elements from file...")
            with open(path, "rb") as file:
                data_dict = pickle.load(file)
            return data_dict['texts']
        else:
            raise ValueError(
                f"Folder output/project/texts does not contain text .pkl dict file with split text size: {self.split_size} and text clean method: {self.clean_meth}. "
                f"Generate it at runtime.")

    def generate_split_texts(self):
        """
        Generate split texts at runtime by reading text bodies from folder with read_input_texts() and
        splitting them in split_size pieces with split_texts(). If clean_meth = "ft", split texts are
        cleaned by function filter_text(). Split texts are returned and stored as .pkl dict in splits_path.

        Returns:
            split_texts (list[str]): list with split texts, so each entry is chunk, sentence or sentence-pair
                                     depending on split_size attribute.

        """
        print("Read text bodies and splitting at runtime...")
        if os.listdir(self.bodies_path):  # Empty folder evaluates to 0
            text_bodies = self.read_input_texts()
            split_texts = self.split_texts(text_bodies)

            with open(os.path.join(self.splits_path, self.texts_split_name), "wb") as file:
                pickle.dump({'texts': split_texts, 'text_split_size': self.split_size}, file, protocol=pickle.HIGHEST_PROTOCOL)

            return split_texts
        else:
            raise ValueError("Folder input/project/text_bodies does not contain any files")

    def read_input_texts(self):
        """
        Read input text bodies saved as .txt from bodies_path folder.

        Returns:
            texts (lst[str]): text bodies

        """
        text_names = sorted([text_file for text_file in os.listdir(self.bodies_path) if text_file.endswith('.txt')])
        print(f'{"Number of texts in folder:":<65}{len(text_names):>10}')

        texts = []
        for text in text_names:
            with open(os.path.join(self.bodies_path, text), "r", encoding="utf-8") as file:
                text_body = file.read()

                # if config.filter_parameters[f"filter_{self.project}"]:
                #     text_body = re.sub(config.filter_parameters[f"filter_pattern_{self.project}"], "", text_body)

            texts.append(text_body)
        return texts

    def split_texts(self, texts):
        """
        Split input text bodies input split_size pieces and filter split texts with
        function filter_texts() if clean_meth = "ft".

        Args:
            texts (list[str]): input text bodies

        Returns:
            list[str]: split (and filtered) text pieces

        """
        if self.split_size == "chunk":
            splits = self.chunk_texts(texts)
        elif self.split_size == "semantic":
            splits = self.semantic_chunk_texts(texts)
        elif self.split_size == "sentence":
            splits = sentencize_text(texts)
        elif self.split_size == "sentence-pairs":
            sentences = sentencize_text(texts)
            splits = pair_sentences(sentences)
        else:
            raise ValueError(
                f"split_size: {self.split_size} is undefined. Valid options are 'chunk', 'sentence', or 'sentence-pairs'.")

        if self.clean_meth == "ft":
            return filter_texts(splits)
        else:
            return splits

    def chunk_texts(self, texts):
        """
        Chunks texts into segments of specified chunk size.

        Args:
            texts (list[str]): input texts

        Returns:
            list[str]: A list with the chunked text segments.
        """
        chunks_out = []
        for text in texts:
            chunks = make_chunks(text, self.chunk_size)
            chunks_out.extend(chunks)
        return chunks_out

    def semantic_chunk_texts(self, texts):
        """
        TODO:
        Add semantic chunking here
        Args:
            texts:

        Returns:

        """
        return 0


    def get_project_name(self):
        path_components = self.bodies_path.split(os.sep)
        return path_components[-2]  # Second to last element of path is project name

    def get_text_bodies_path(self):
        return self.bodies_path
