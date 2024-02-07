import os
import pickle
from sentence_transformers import SentenceTransformer

import config

class EmbeddingsPreProcess:

    def __init__(self, emb_from_file, emb_path, clean_meth, split_size):
        """
        Class EmbeddingsPreProcess stores the variables needed to initialize the embeddings: emb_from_file,
        embeddings path, text clean-method, and text split-size. It also handles the functionality to
        initialize embeddings: loading pre-trained embeddings from file or generating them at runtime
        based on input text data.

        Parameters:
            emb_from_file (bool): Boolean indicator whether to load existing embeddings from file or generate them
                                  at runtime from text data.
            emb_path (str): Directory path where embeddings are stored.
            clean_meth (str): Text clean method. (def for default, ft for filter-text function,
                              vect for vectorization param in BERTopic)
            split_size (str): Text split size. (chunk, sentence, or sentence-pairs)

        Attributes:
            embedding_name (str): embedding filename based on split_size and clean_method.
            bert_model (str): pre-trained sentence BERT model name, defined in config.
        """

        self.emb_from_file = emb_from_file
        self.path = emb_path
        self.clean_meth = clean_meth
        self.split_size = split_size
        self.embedding_name = f"embeddings_{split_size}_{clean_meth}.pkl"
        self.bert_model = config.emb_pp_params["bert_model"]

    def get_embeddings(self, data):
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
            embeddings = self.generate_embeddings(data)
        print(f'{"Embeddings shape:":<65}{str(embeddings.shape):>10}')
        return embeddings

    def load_embeddings(self):
        """
        Load embeddings saved as .pkl dict file saved in folder self.path.

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        path = os.path.join(self.path, self.embedding_name)
        if os.path.exists(path):
            print(f"Embedding file name: {self.embedding_name}. Reading embeddings from file...")
            with open(os.path.join(self.path, self.embedding_name), "rb") as file:
                data_dict = pickle.load(file)
            return data_dict['embeddings']
        else:
            raise ValueError(
                f"Folder output/project/embeddings does not contain emb .pkl dict file with split text size: {self.split_size} and text clean method: {self.clean_meth}. "
                f"Generate it at runtime.")

    def generate_embeddings(self, data):
        """
        Generate embeddings from input text data using SentenceTransformer module.
        Return embeddings and save in folder self.path the embeddings, text_split_size,
        and bert_model as .pkl dict. Sentence BERT model name given by self.bert_model.

        Args:
            data (list[str]): input text data

        Returns:
            embeddings (torch.Tensor): text embeddings, each doc as a 768-dim vector. Shape: (num docs, 768)
        """
        print("Initializing embeddings at runtime...")
        model = SentenceTransformer(self.bert_model)
        embeddings = model.encode(data, show_progress_bar=True)

        with open(os.path.join(self.path, self.embedding_name), "wb") as file:
            pickle.dump({'embeddings': embeddings, 'text_split_size': self.split_size, 'bert_model': self.bert_model}, file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return embeddings

