import os
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd


def read_input_texts(data_path):
    text_names = sorted([text_file for text_file in os.listdir(data_path) if text_file.endswith('.txt')])
    print(f'{"Number of texts in folder:":<65}{len(text_names):>10}')

    texts = []
    for text in text_names:
        with open(os.path.join(data_path, text), "r", encoding="utf-8") as file:
            text_body = file.read()
        texts.append(text_body)
    return texts

def sentencize_text(texts):
    return [sentence for t in texts for sentence in t.split(".")]


if __name__ == "__main__":


    pass





















