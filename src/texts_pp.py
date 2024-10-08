import os
from typing import List
from nltk.tokenize import word_tokenize
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

def make_chunks_token(text, max_length):
    """
    Chunk individual text to pieces of max length

    Args:
        text (str)
        max_length (int)

    Returns:
        chunks (list[str]): text in chunks
    """
    chunks = []
    tokenized_sentence = word_tokenize(text) 
    while tokenized_sentence:
        chunk = " ".join(tokenized_sentence[:max_length])
        chunks.append(chunk)
        tokenized_sentence = tokenized_sentence[max_length:]
    return chunks

def truncate_sentences(sentences: List[str], max_len: int = 512) -> List[str]:
    """
    Truncate sentences to max length

    Returns:
        List[str]: truncated sentences
    """
    return [i for s in sentences for i in make_chunks(s, max_len)]

def truncate_sentences_token(sentences: List[str], max_len: int = 512) -> List[str]:
    """
    Truncate sentences to max length

    Returns:
        List[str]: truncated sentences
    """
    return [i for s in sentences for i in make_chunks_token(s, max_len)]

def sentencize_text(texts):
    """
    TODO:
    - Use Spacy Sentencizer to make sentences, splitting on .?! is imprecise

    Splits input texts into sentences, keep sentences of each document in a separate list

    Args:
        texts (List[str]): text bodies

    Returns:
        List[List[str]]: A list of list with sentences for each input text
    """
    delimiters = ['.', '?', '!']
    pattern = '\s|'.join(map(re.escape, delimiters))  # Add whitespace \s, regex OR delim |, and add backslash with escape
    return [re.split(pattern, text) for text in texts]

def pair_sentences(text_sentences):
    """
    Pair sentences together

    Args:
        text_sentences (List[List[str]]): lst of lst with sentence of each document

    Returns:
        List[str]: paired sentences
    """
    sentence_pairs = []
    for text in text_sentences:
        pairs = [(text[i]+". "+text[i+1] if i+1 < len(text) else text[i]) for i in range(0, len(text), 2)]
        sentence_pairs.append(pairs)
    return sentence_pairs

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
        print(path)
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
        elif self.split_size == "chunk_len":
            text_sentences = sentencize_text(texts)
            print(text_sentences)
            splits = self.chunk_sents_len(text_sentences)
        elif self.split_size == "sentence":
            splits = sentencize_text(texts)
        elif self.split_size == "sentence-pairs":
            text_sentences = sentencize_text(texts)
            splits = pair_sentences(text_sentences)
        elif self.split_size == "tokenize":
            text_sentences = sentencize_text(texts)
            splits = self.chunk_sents_len_on_tokens(text_sentences)
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

    def chunk_sents_len(self, text_sentences: List[List[str]]):
        """
        Chunk sentences together in chunks of at most max_len length

        Args:
            text_sentences: sentences, which is the text split on "."

        Returns:
            chunks (List[List[str]]): sentence chunks
        """
        chunks = []
        text_sentences = text_sentences[:5]
        for sentences in text_sentences:
            print(sentences)
            trunc_sentences = truncate_sentences(sentences, self.chunk_size)
            lens = [len(s) for s in trunc_sentences]  # Sentence lengths
            sent_indices = self.get_sent_chunk_inds(lens)  # Sentence indices to chunk together
            chunks_text = [". ".join([trunc_sentences[i] for i in inds]) for inds in sent_indices]  # Make chunks from the sentence indices
            chunks.extend(chunks_text)
        return chunks

    def chunk_sents_len_on_tokens(self, text_sentences: List[List[str]]):
        """
        Chunk sentences together in chunks of at most max_len length tokens

        Args:
            text_sentences: sentences, which is the text split on "."

        Returns:
            chunks (List[List[str]]): sentence chunks
        """
        chunks = []
        for sentences in text_sentences:
            trunc_sentences = truncate_sentences_token(sentences, self.chunk_size)
            trunc_sentences_token = [word_tokenize(t) for t in trunc_sentences] 
            lens = [len(s) for s in trunc_sentences_token]  # Sentence based on tuncated sentences
            sent_indices = self.get_sent_chunk_inds(lens)  # Sentence indices to chunk together
            chunks_text = [". ".join([trunc_sentences[i] for i in inds]) for inds in sent_indices]  # Make chunks from the sentence indices
            chunks.extend(chunks_text)
        return chunks

    def get_sent_chunk_inds(self, lens: List[int]):
        """
        Find indices of sentences to chunk together to from sentence chunks of at most max_len length

        Returns:
            List[List[int]]: Lst of lst with sentence indices to be chunked
        """
        cumsum = 0  # Cumulative sum
        inds = []  # Indices for a chunk
        chunk_inds = []  # Lst of lst with indices

        for i, l in enumerate(lens):
            if cumsum + l < self.chunk_size:
                cumsum += l
                inds.append(i)
            else:
                if inds:
                    chunk_inds.append(inds)  # Only add to output lst if inds non-empty
                if l >= self.chunk_size:
                    chunk_inds.append([i])  # The case where l itself is > max_len
                cumsum = l if l < self.chunk_size else 0
                inds = [i] if l < self.chunk_size else []
            if i == len(lens) - 1:  # Add the rest indices to chunk_inds output
                chunk_inds.append(inds)

        return chunk_inds

    def get_project_name(self):
        path_components = self.bodies_path.split(os.sep)
        return path_components[-2]  # Second to last element of path is project name

    def get_text_bodies_path(self):
        return self.bodies_path