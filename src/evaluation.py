import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
# from octis.evaluation_metrics.diversity_metrics import TopicDiversity

class Evaluation:

    def __init__(self):
        print("\nEval object created")

    def calculate_coherence(self, topic_model, texts):
        print("Calculating coherence...")
        topics = np.array(topic_model.topics_)  # Topic id for each document, [-1, -1, 58, 17, 23, -1, ...]
        documents = pd.DataFrame({"Document": texts,
                                  "ID": range(len(texts)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                       for topic in range(len(set(topics)) - 1)]

        # Evaluate
        # For c_npmi, values range between [-1, 1] where 1 indicates perfect association. Reasonably good values are between 0.15-0.20,
        # based on the BERTopic paper.
        # For c_v, values range between [0, 1] where 1 indicates perfect association. Reasonably good performance is coherence > 0.5 (from
        # Computational Social Science Book)
        coherence_metrics = ['c_v', 'c_npmi']
        dict = {}

        for metric in coherence_metrics:
            coherence_model = CoherenceModel(topics=topic_words,
                                             texts=tokens,
                                             corpus=corpus,
                                             dictionary=dictionary,
                                             coherence=metric)
            coherence = coherence_model.get_coherence()
            print(f"Coherence score {metric}: {coherence:.4f}")
            dict[metric] = coherence

        return dict

    def calculate_diversity(self, topic_model):
        # words_output = []  # 2-D list with topic words
        # for id, distribution in topic_model.get_topics().items():
        #     words = [tuple_[0] for tuple_ in distribution]
        #     words_output.append(words)
        #
        # model_output = {"topics": words_output[1:]}
        #
        # metric = TopicDiversity(topk=5)  # Initialize metric
        # topic_diversity_score = metric.score(model_output)  # Compute score of the metric
        # return topic_diversity_score
        pass

