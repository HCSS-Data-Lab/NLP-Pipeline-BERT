from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import time
import os

import config

class Analysis:

    def __init__(self, data, models_path, model_from_file="True", clean_meth="def"):
        """
        Class Analysis handles running the topic modeling analysis with the BERTopic module.

        It loads a pre-trained topic-model from file or initializes a topic-model object from the
        BERTopic module at runtime, using the input text data.

        Parameters:
            data (list[str]): input text data
            models_path (str): path where models are stored or retrieved
            model_from_file (bool): use model saved in file or not
            clean_meth (str): text cleaning method

        Attributes:
            data (list[str]):
            models_path (str):
            model_from_file (bool):
            clean_meth:
            merged (bool): Indicator to show whether topic output has been merged or not
        """

        self.data = data
        self.models_path = models_path
        self.model_from_file = model_from_file
        self.clean_meth = clean_meth
        self.merged = False

    # def initialize_topic_model(self):
    #     model_name = f"bertopic_model_{self.text_clean}"
    #     if self.model_from_file:
    #         print(f"Topic-model name: {model_name}\nReading topic-model from file...")
    #         start = time.time()
    #         topic_model = BERTopic.load(os.path.join(self.models_path, model_name))
    #         print(f"Time elapsed to load topic-model: {time.time() - start:.3f} seconds")
    #         return topic_model
    #
    #     else:
    #         print("Initializing topic-model at runtime...")
    #         if self.text_clean == "vect":
    #             print("Initializing topic model with stop words vectorizer...")
    #             vectorizer_model = CountVectorizer(stop_words="english")
    #             topic_model = BERTopic(vectorizer_model=vectorizer_model,
    #                                    embedding_model=config.)
    #
    #         else:
    #             topic_model = BERTopic(embedding_model=embedding_model)
    #
    #         topics, probs = topic_model.fit_transform(data)
    #
    #         # Saving topic_model
    #         topic_model.save(os.path.join(drive_path, model_name))
