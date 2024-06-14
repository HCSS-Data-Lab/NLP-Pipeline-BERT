import os
import pickle
from sentence_transformers import SentenceTransformer
# from FlagEmbedding import BGEM3FlagModel
from multiprocessing import Pool
import numpy as np
from itertools import repeat
import umap.umap_ as umap

import config

def encode_chunk(chunk, model):
    return model.encode(chunk, show_progress_bar=True)

class EmbeddingsPreProcess:

    def __init__(self, emb_path):
        """
        Class EmbeddingsPreProcess stores the variables needed to initialize the embeddings: emb_from_file,
        embeddings path, text clean-method, and text split-size. It also handles the functionality to
        initialize embeddings: loading pre-trained embeddings from file or generating them at runtime
        based on input text data. It handles the same functionality for reduced embeddings: loading reduced
        embeddings from file or at runtime, so mapping the embeddings to 2 dimensions using UMAP.

        Parameters:
            emb_path (str): Directory path where embeddings are stored.

        Attributes:
            emb_from_file (bool): Boolean indicator whether to load existing embeddings from file or generate them
                                  at runtime from text data.
            red_rom_file (bool): Boolean indicator whether to load existing reduced embeddings from file or generate them
                                  at runtime from embeddings.
            clean_meth (str): Text clean method. (def for default, ft for filter-text function,
                              vect for vectorization param in BERTopic)
            split_size (str): Text split size. (chunk, sentence, or sentence-pairs)
            chunk_size (str): Chunk size in number of characters
            bert_model (str): Pre-trained sentence BERT model name, defined in config.
            embedding_name (str): Embedding filename based on split_size and clean_method.
            red_emb_name (str): Reduced embeddings filename.
            random_state (int): Random state for reproducability, using in UMAP
        """
        self.emb_from_file = config.LOAD_EMBEDDINGS_FROM_FILE
        self.red_from_file = config.LOAD_REDUCED_EMBEDDINGS_FROM_FILE

        self.path = emb_path

        self.bert_model = config.model_parameters["emb_model"]
        self.bert_model_str = self.bert_model.split("/")[-1]
        self.split_size = config.text_splitting_parameters["split_size"]
        self.chunk_size = config.text_splitting_parameters["chunk_size"]

        self.embedding_name = f"embeddings_{self.split_size}{self.chunk_size}_{self.bert_model_str}.pkl"
        self.red_emb_name = f"red_embeddings_{self.split_size}{self.chunk_size}_{self.bert_model_str}.npy"

        self.clean_meth = config.tm_parameters["clean_meth"]
        self.random_state = config.umap_parameters["random_state"]

    def get_embeddings(self, data, parallel=False):
        """
        Get embeddings by loading them from file with load_embeddings() or generate
        them at runtime based on input text data with generate_embeddings().

        Args:
            data (list[str]): input text data

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        if self.emb_from_file:
            embeddings = self.load_embeddings()
        else:
            if parallel:
                embeddings = self.generate_embeddings_parallel(data)
            else:
                embeddings = self.generate_embeddings(data)
        print(f'{"Embeddings shape:":<65}{str(embeddings.shape):>10}\n')
        return embeddings

    def load_embeddings(self):
        """
        Load embeddings saved as .pkl dict file saved in folder self.path.

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        path = os.path.join(self.path, self.embedding_name)
        if os.path.exists(path):
            print(f"Embedding file name: {self.embedding_name}. \nReading embeddings from file...")
            with open(os.path.join(self.path, self.embedding_name), "rb") as file:
                data_dict = pickle.load(file)
            return data_dict['embeddings']
        else:
            raise ValueError(
                f"Folder output/project/year/embeddings does not contain specified emb .pkl dict file. "
                f"Set LOAD_EMBEDDINGS_FROM_FILE to False and generate embeddings at runtime.")

    def generate_embeddings(self, data):
        """
        Generate embeddings from input text data using SentenceTransformer module.
        Return embeddings and save in folder self.path the embeddings, text_split_size,
        and bert_model as .pkl dict. Sentence BERT model name given by self.bert_model.

        Args:
            data (List[str]): input text data (chunks)

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        print("Initializing embeddings at runtime...")
        if config.model_parameters['non_st_model']:
            # IMPLEMENT HERE
            pass
            # model = BGEM3FlagModel(self.bert_model,  use_fp16=True)
            # embeddings = model.encode(data)
        else:
            model = SentenceTransformer(self.bert_model, trust_remote_code=True)
            embedding_dim = model.get_sentence_embedding_dimension()
            max_seq_length = model.max_seq_length
            print(f"Embedding dimension: {embedding_dim}\nMax sequence length: {max_seq_length}")
            embeddings = model.encode(data, show_progress_bar=True)

        with open(os.path.join(self.path, self.embedding_name), "wb") as file:
            pickle.dump({'embeddings': embeddings, 'text_split_size': self.split_size, 'bert_model': self.bert_model}, file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

    def generate_embeddings_parallel(self, data):
        print("Initializing embeddings in parallel...")
        model = SentenceTransformer(self.bert_model)

        num_processes = 4
        process_chunks = [data[i::num_processes] for i in range(num_processes)]

        with Pool() as p:
            results = p.starmap(encode_chunk, zip(process_chunks, repeat(model)))

        embeddings = np.concatenate(results, axis=0)

        emb_name = self.embedding_name[:-4]+"_pT.pkl"  # Add 'pT' as parallel=True flag to embedding name
        with open(os.path.join(self.path, emb_name), "wb") as file:
            pickle.dump({'embeddings': embeddings, 'text_split_size': self.split_size, 'bert_model': self.bert_model},
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

    def get_red_embeddings(self, embeddings):
        """
        Get reduced embeddings by loading them from file with load_red_embeddings() or generate
        them at runtime based on input embeddings with generate_red_embeddings().

        Args:
            embeddings (torch.Tensor): high-dimensional text embeddings

        Returns:
            red_embeddings (np.array): 2-dim embeddings as numpy array
        """
        if self.red_from_file:
            red_embeddings = self.load_red_embeddings()
        else:
            red_embeddings = self.generate_red_embeddings(embeddings)
        print(f'{"Reduced Embeddings shape:":<65}{str(red_embeddings.shape):>10}\n')
        return red_embeddings

    def load_red_embeddings(self):
        """
        Load reduced embeddings saved as np array in folder self.path.

        Returns:
            np.array: reduced embeddings. Shape: (num docs, 2)
        """
        print(f"Reduced embedding file name: {self.red_emb_name}. \nReading reduced embeddings from file...")
        return np.load(os.path.join(self.path, self.red_emb_name))

    def generate_red_embeddings(self, embeddings):
        """
        Generate reduced embeddings from text embeddings by mapping to 2 dims using UMAP.

        Args:
            embeddings (torch.Tensor): text embeddings

        Returns:
            reduced_embeddings (np.array): reduced embeddings. Shape: (num docs, 768)
        """
        print("Initializing reduced embeddings at runtime...")
        reduced_embeddings = umap.UMAP(n_neighbors=15,
                                       n_components=5,
                                       min_dist=0.0,
                                       metric='cosine',
                                       random_state=self.random_state).fit_transform(embeddings)

        # Saving reduced embeddings to file
        np.save(os.path.join(self.path, self.red_emb_name), reduced_embeddings)
        return reduced_embeddings
