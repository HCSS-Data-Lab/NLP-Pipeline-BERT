import os
import numpy as np
import time
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
from typing import List

import config
from utils.visualize_func import visualize_topics_over_time_

def read_text_body(folder: str, name: str) -> str:
    with open(os.path.join(folder, name), encoding="utf-8") as file:
        text_body = file.read()
    return text_body


def find_phrases(keywords):
    """
    Find phrases in keywords, ie keywords that consist of two terms, like 'law enforcement'; identified by space
    """
    return [kw for kw in keywords if " " in keywords]


def preprocess_texts(texts, phrases):
    """
    Preprocess texts to replace spaces in specified phrases with underscores.
    """
    preprocessed_texts = {}
    for filename, content in texts.items():
        for phrase in phrases:
            content = content.replace(phrase, phrase.replace(" ", "_"))
        preprocessed_texts[filename] = content
    return preprocessed_texts


def compute_tf_idf_scores(texts, keywords):
    """
    Computes TF-IDF scores for specific keywords in a collection of documents.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts.values())

    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=texts.keys(),
                            columns=tfidf_vectorizer.get_feature_names_out())

    # Adjust keywords for phrases (replace spaces with underscores)
    adjusted_keywords = [keyword.replace(" ", "_") for keyword in keywords]
    df_keywords_tfidf = df_tfidf.loc[:, df_tfidf.columns.intersection(adjusted_keywords)]

    return df_keywords_tfidf


def aggregate_scores(tf_idf_scores):
    """
    Aggregates TF-IDF scores by summing them up for each document.

    Parameters:
    - tf_idf_scores: DataFrame containing the TF-IDF scores for specified keywords across documents.

    Returns:
    A Series with the sum of TF-IDF scores for each document.
    """
    # Summing up the TF-IDF scores for each document
    aggregated_scores = tf_idf_scores.sum(axis=1)
    return aggregated_scores

def extract_date(name: str):
    # Regular expression to match the date pattern
    match = re.search(r'\d{4}-\d{2}-\d{2}', name)
    if match:
        return match.group()
    return None

class DynamicTopicModeling:

    def __init__(self, project_root, dataset_name):
        self.project_root = project_root
        self.dataset_name = dataset_name
        self.project_folder = os.path.join(self.project_root, "input", self.dataset_name)
        self.topics_ot_from_file = config.LOAD_TOPICS_OVER_TIME_FROM_FILE

    def get_time_stamps(self, texts: dict):
        """
        Find time stamps for input texts

        Args:
            texts (dict):

        Returns:
            time_stamps (List[tuple[str, str]]):
        """
        time_stamps = []
        for text_name, chunks in texts.items():
            date = extract_date(text_name)
            if date:
                for chunk in chunks:
                    time_stamps.append({"date": date, "chunk": chunk})
            else:
                raise ValueError("No date was recognized in the text name. Make sure a date in the format year-month-day is part of the text name.")
        return time_stamps

    # def get_topics_over_time(self, output_folder):
    #     if self.topics_ot_from_file:
    #         topics_over_time = pd.read_csv(os.path.join(output_folder, "models", "topics_over_time.csv"))
    #     else:
    #         self.generate_topics_over_time()

    def run_dtm(self, topic_model, text_chunks, timestamps, nr_bins, output_folder, save_topics=True):
        print("Generating topics over time...")
        start = time.time()
        topics_over_time = topic_model.topics_over_time(text_chunks, timestamps, datetime_format="%Y-%m-%d", nr_bins=nr_bins)
        print(f"Time elapsed for topics over time: {time.time() - start:.4f} seconds\n")

        if save_topics:
            topics_over_time.to_csv(os.path.join(output_folder, "models", "topics_over_time.csv"), index=False)

        return topics_over_time

    def visualize_topics(self, topic_model, topics_over_time, output_folder, year_str, top_n=None, topics_to_show=None,
                         topics_background=None, background_alpha=None, legend_opaque=True):
        print("Visualizing topics over time...")
        custom_labels_df = pd.read_csv(os.path.join(output_folder, "models", "Topic_Descriptions.csv"))
        custom_labels = custom_labels_df["Topic Name"].to_list()
        topic_model.set_topic_labels(custom_labels)
        fig = visualize_topics_over_time_(topic_model,
                                          topics_over_time,
                                          top_n_topics=top_n,
                                          topics=topics_to_show,
                                          normalize_frequency=True,
                                          custom_labels=True,
                                          topics_background=topics_background,
                                          background_alpha=background_alpha,
                                          color_legend_opaque=legend_opaque,
                                          title="<b>Trendanalyse securitisering 2015 - 2022</b>")

        out_path = os.path.join(output_folder, "figures", f"topics_over_time_{year_str}.html")
        if os.path.exists(out_path):
            fig.write_html(os.path.join(output_folder, "figures", f"test_topics_over_time_{year_str}.html"))
        else:
            fig.write_html(out_path)





