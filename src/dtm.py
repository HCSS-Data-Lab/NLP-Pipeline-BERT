import os
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
from typing import List

import config

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

    def __init__(self, project_root, project, dtm_years=None):
        self.project_root = project_root
        self.project = project
        self.project_folder = os.path.join(self.project_root, "input", self.project)
        self.dtm_years = dtm_years
        self.sample = config.dtm_parameters["sample_size"]

    def sample_copy_docs(self):
        """
        Sample and copy docs
        """
        print("Sampling documents...")
        for year in self.dtm_years:
            print(f"Sampling for year: {year}", flush=True)

            # Get samples
            samples = self.sample_docs(year)

            # Save to output folder
            output_path = os.path.join(self.project_folder, f"{year}_s{self.sample}", "raw_texts")
            os.makedirs(output_path, exist_ok=True)
            self.copy_names(samples, year, output_path)

    def sample_docs(self, year):
        """
        Sample self.sample docs from /input_folder/[year]

        Args:
            year (str): year folder to take samples from

        Returns:
            List[str]: sampled text names (so not full paths)
        """
        input_path = os.path.join(self.project_folder, year, "raw_texts")
        text_names = sorted([text_file for text_file in os.listdir(input_path) if text_file.endswith('.txt')])
        sample_inds = np.random.choice(len(text_names), size=int(self.sample * len(text_names)), replace=False)
        return [text_names[i] for i in sample_inds]

    def find_keyword_docs(self, year_str, keywords):
        """
        Find docs with keywords
        """
        output_folder = os.path.join(self.project_folder, year_str, "text_bodies")
        os.makedirs(output_folder, exist_ok=True)

        for year in self.dtm_years:
            print(f"Finding docs for {year}...")
            input_path = os.path.join(self.project_folder, year, "text_bodies")
            text_names = sorted([text_file for text_file in os.listdir(input_path) if text_file.endswith('.txt')])
            for name in text_names:
                text_body = read_text_body(input_path, name)
                unique_words = set(text_body.split())
                if any(s in unique_words for s in keywords):
                    shutil.copy(os.path.join(self.project_folder, year, "text_bodies", name), os.path.join(output_folder, name))

        print(f"Files copied: {len(os.listdir(output_folder))}")

    def find_keyword_docs_tfidf(self, keywords, keyword_year_str):
        """
        Find docs most relevant to the keywords based on tf-idf calculation and copy to output folder
        """
        # Make output folder
        output_folder = os.path.join(self.project_folder, keyword_year_str, "text_bodies")
        os.makedirs(output_folder, exist_ok=True)

        # Find phrases, ie keywords with two or more terms
        phrases = find_phrases(keywords)

        threshold = config.dtm_parameters["tfidf_threshold"]
        threshold_type = config.dtm_parameters["tfidf_threshold_type"]

        # Find most relevant documents for each year, and copy to the output folder
        for year in self.dtm_years:
            # Find relevance score (tf-idf) for each document
            texts = self.load_texts_from_directory(year)
            preprocessed_texts = preprocess_texts(texts, phrases)
            tf_idf_scores = compute_tf_idf_scores(preprocessed_texts, keywords)
            aggregated_scores = aggregate_scores(tf_idf_scores)
            sorted_scores = aggregated_scores.sort_values(ascending=False)

            if threshold_type == "document":
                self.copy_docs_doc_threshold(sorted_scores, threshold, year, output_folder)

            elif threshold_type == "value":
                self.copy_docs_value_threshold(sorted_scores, threshold, year, output_folder)

            else:
                raise ValueError(f"The given threshold type {threshold_type} is invalid, options are 'document' or 'value'")

    def load_texts_from_directory(self, year):
        """
        Loads and reads text files from a specified directory, and prints the names of the documents.

        Args:
            year (str): year

        Returns:
            dict: dict where keys are filenames and values are the content of the files.
        """
        directory_path = os.path.join(self.project_folder, year, "text_bodies")
        texts = {}
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            print(f"The directory {directory_path} does not exist.")
            return texts

        # Iterate through all the files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            # Ensure we're only reading text files
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts[filename] = file.read()
        return texts

    def copy_docs_doc_threshold(self, sorted_scores, threshold, year, output_folder):
        # Find text names until the threshold
        doc_ind_threshold, cumsum = self.find_cumsum_value(sorted_scores, threshold)
        names_until_threshold = sorted_scores.index[:doc_ind_threshold]

        # Copy to output folder
        self.copy_names(names_until_threshold, year, output_folder)
        print(f"Year: {year} -- {len(names_until_threshold)} ({threshold * 100:.0f}%) documents contain {cumsum / np.sum(sorted_scores) * 100:.2f}% of the total tf-idf value")

    def copy_docs_value_threshold(self, sorted_scores, threshold, year, output_folder):
        # Find text names until the value threshold
        name, perc_thresh = self.find_perc_threshold_name(sorted_scores, threshold)
        names_until_threshold = self.get_names_until_treshold(sorted_scores, name)

        # Copy to output folder
        self.copy_names(names_until_threshold, year, output_folder)
        print(f"Year: {year} -- {threshold * 100:.0f}% of relevance value (tf-idf) is contained in {len(names_until_threshold)} ({perc_thresh:.2f}%) documents")

    def find_cumsum_value(self, sorted_scores: pd.Series, percent_threshold: float):
        """
        Calculate the cumulative tf-idf value for the top x% of documents in a sorted series.

        Args:
            sorted_scores (pd.Series): A pandas Series where the index contains document names
            and the values are tf-idf scores, sorted in descending order.
            percent_threshold (float): The top percentage of documents for which to
            calculate the cumulative tf-idf value.

        Returns:
            doc_ind_threshold (int): document index of the threshold
            cumulative_value (float): cumulative tf-idf value
        """
        # Calculate the number of documents that constitute the top 20%
        total_documents = len(sorted_scores)
        doc_ind_threshold = int(total_documents * percent_threshold)

        # If doc_ind_threshold is 0 (in case of very small datasets), take at least one document
        doc_ind_threshold = max(doc_ind_threshold, 1)

        # Summing the tf-idf values of the top x% documents
        cumulative_value = sorted_scores.iloc[:doc_ind_threshold].sum()
        return doc_ind_threshold, cumulative_value

    def find_perc_threshold_name(self, sorted_scores: pd.Series, value_threshold_perc: float):
        """
        Find the percentage of documents that cumulatively contain a specified percentage
        of the total tf-idf value in a sorted series, and the name of the document at
        the threshold.

        Args:
            sorted_scores (pd.Series): A pandas Series where the index contains document names
            and the values are tf-idf scores, sorted in descending order of relevance.
            value_threshold_perc (float): The target cumulative percentage of the total tf-idf value
            to find (e.g., 0.8 for 80%).

        Returns:
            float: The percentage of documents that cumulatively reach the specified tf-idf
            value threshold.
        """
        cumsum = sorted_scores.cumsum()
        value_threshold = value_threshold_perc * sorted_scores.sum()  # Threshold value, not percentage

        doc_name_threshold = cumsum[cumsum >= value_threshold].index[0]  # Document name that is the threshold

        # Find perc of this doc in all docs
        docs_until_threshold = sorted_scores.index.get_loc(doc_name_threshold) + 1
        perc_of_docs = docs_until_threshold / len(sorted_scores) * 100
        return doc_name_threshold, perc_of_docs

    def get_names_until_treshold(self, sorted_scores: pd.Series, name: str) -> pd.Index:
        """
        Extracts and returns the names (index labels) of documents from the beginning of a
        sorted pandas Series up to and including a specified document name.

        Args:
            sorted_scores (pd.Series): A pandas Series with document names as the index and
            a metric (e.g., tf-idf scores) as the values, sorted in descending order of the metric.
            name (str): The name of the document up to which names are to be returned. This name
            must exist in the Series' index.

        Returns:
            pd.Index: An Index object containing the names of the documents up to and including
            the specified name.
        """
        position = sorted_scores.index.get_loc(name)
        names_until_threshold = sorted_scores.index[:position + 1]
        return names_until_threshold

    def copy_names(self, text_names, year, output_folder):
        """
        Copy the files in text_names to the output folder
        """
        for name in text_names:
            shutil.copy(os.path.join(self.project_folder, year, "text_bodies", name), os.path.join(output_folder, name))

    def get_time_stamps(self, texts: dict):
        """
        Find time stamps for input texts

        Args:
            texts (dict): texts dictionary with text names (keys) and list of chunks (values)

        Returns:
            time_stamps (lst):
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














