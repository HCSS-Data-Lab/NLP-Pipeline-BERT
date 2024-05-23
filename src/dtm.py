import os
import numpy as np
import time
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
from typing import List, Dict

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

    def __init__(self, project_root, dataset_name, **kwargs):
        self.project_root = project_root
        self.dataset_name = dataset_name
        self.project_folder = os.path.join(self.project_root, "input", self.dataset_name)
        self.topics_ot_from_file = config.LOAD_TOPICS_OVER_TIME_FROM_FILE

    def get_time_stamps(self, texts: dict):
        """
        Find time stamps for input texts

        Args:
            texts (Dict[tuple[str, List[str]]]): dict with tuples of (text id, list of chunks)

        Returns:
            timestamps (List[str]): only timestamps
            timestamp_chunks (List[Dict[str, str]]): timestamp and chunk in tuple
        """
        timestamps = []
        timestamp_chunks = []
        for text_name, chunks in texts.items():
            date = extract_date(text_name)
            if date:
                for chunk in chunks:
                    timestamps.append(date)
                    timestamp_chunks.append({"date": date, "chunk": chunk})
            else:
                raise ValueError("No date was recognized in the text name. Make sure a date in the format year-month-day is part of the text name.")
        return timestamps, timestamp_chunks

    def get_time_stamps_speeches(self, speeches: List[Dict[str, str]]):
        """
        Find time stamps for input speeches

        Args:
            speeches (List[Dict[str, str]]): list of dict with speeches as {speech id, speech text}

        Returns:
            timestamps (List[str]): only timestamps
            timestamp_chunks (List[Dict[str, str]]): timestamp and chunk in tuple
        """
        timestamps = []
        timestamp_chunks = []
        for speech in speeches:
            speech_id, speech_text = speech.values()
            date = extract_date(speech_id)
            if date:
                timestamps.append(date)
                timestamp_chunks.append({"date": date, "speech": speech_text})
            else:
                raise ValueError("No date was recognized in the text name. Make sure a date in the format year-month-day is part of the text name.")
        return timestamps, timestamp_chunks

    def get_timestamp_bins(self, timestamps, frequency='QS'):
        """
        Generate timestamp bins for given timestamps, which is the date str for each text chunk,
        and assign them to bins with size frequency. E.g. if frequency='Q', the size is a quarter;
        or if frequency='M', the size is a month. 'QS' means quarter start, so the label is the start of the quarter, not end.

        Args:
            timestamps (List[str]): timestamps as list of date str
            frequency (str): size of timestamp bin, see pandas.date_range documentation for options

        Returns:
            timestamp_bins (pandas.Categorical): each timestamp assigned the correct label of its timestamp bin
        """
        timestamps = pd.to_datetime(timestamps, format="%Y-%m-%d")

        # Define the start and end of your period
        start_year = timestamps.min().year
        end_year = timestamps.max().year

        # Generate bins
        quarters = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq=frequency)

        # Bin each timestamp into a fraction of size frequency
        timestamp_bins = pd.cut(timestamps, bins=quarters, labels=quarters[:-1].strftime('%Y-%m-%d'),
                                include_lowest=True)
        return timestamp_bins

    def run_dtm(self, topic_model, text_chunks, timestamps, output_folder, save_topics=True):
        print("Generating topics over time...")
        start = time.time()
        topics_over_time = topic_model.topics_over_time(text_chunks, timestamps, datetime_format="%Y-%m-%d")
        print(f"Time elapsed for topics over time: {time.time() - start:.4f} seconds\n")

        if save_topics:
            topics_over_time.to_csv(os.path.join(output_folder, "models", "topics_over_time.csv"), index=False)

        return topics_over_time

    def visualize_topics(self, topic_model, topics_over_time, output_folder, year_str, use_custom_labels=False, custom_vis_func=False):
        print("Visualizing topics over time...")

        if use_custom_labels:
            custom_labels_df = pd.read_csv(os.path.join(output_folder, "models", "Topic_Descriptions.csv"))
            custom_labels = custom_labels_df["Topic Name"].to_list()
            topic_model.set_topic_labels(custom_labels)

        if custom_vis_func:
            fig = visualize_topics_over_time_(topic_model,
                                              topics_over_time,
                                              **config.dtm_plotting_parameters)
        else:
            fig = topic_model.visualize_topics_over_time(topics_over_time,
                                                         **config.dtm_plotting_parameters)

        out_path = os.path.join(output_folder, "figures", f"topics_over_time_{year_str}.html")
        if os.path.exists(out_path):
            fig.write_html(os.path.join(output_folder, "figures", f"test_topics_over_time_{year_str}.html"))
        else:
            fig.write_html(out_path)





