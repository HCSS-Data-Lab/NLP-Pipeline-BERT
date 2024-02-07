import os
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import matplotlib.pyplot as plt







if __name__ == "__main__":

    path = r"C:\Users\ArneEichholtz\PycharmProjects\NLP-Pipeline-BERT\output\Politie\embeddings\embeddings_chunk_def.pkl"

    if os.path.exists(path):
        print("path exists")
    else:
        print("does not exist")





























