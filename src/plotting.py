import numpy as np
import os

from utils.visualize_func import get_sample_indices
from utils.visualize_func import visualize_documents_
from utils.text_process_llm import get_summary_sampled_docs
from utils.text_process_llm import get_summary_labels

import config

def first_at_end(lst):
    """Append the first element of list at the end"""
    lst.append(lst.pop(0))
    return lst

def get_num_docs_topic(hover_labels):
    """
    Find number of docs (text pieces) classified to a topic; the rest is classified to the 'outlier' topic.

    Args:
        hover_labels:

    Returns:
        int: Number of docs

    """
    return len(hover_labels) - hover_labels.count("")

class Plotting:

    def __init__(self, topic_model, reduced_embeddings, model_name, docs, folder="", save_html=False, merged=False, plot_non_docs=False, summarize_labels=False, summarize_docs=False, rag=None):
        # Parameters
        self.topic_model = topic_model
        self.red_emb = reduced_embeddings
        self.model_name = model_name
        self.docs = docs
        self.num_docs = len(docs)
        self.folder = folder
        self.save_html = save_html
        self.merged = merged
        self.summarize_labels = summarize_labels
        self.summarize_docs = summarize_docs
        self.RAG=rag

        # Variables from topic-model object
        self.topics = self.get_topics()
        self.num_topics = self.get_num_topics()

        # Plotting parameters
        self.n_total = config.plotting_parameters["n_total"]
        self.sample = config.plotting_parameters["sample"]
        self.RAG_n_words_legend = config.rag_parameters["RAG_n_words_legend"]
        self.n_words_legend = config.plotting_parameters["n_words_legend"]
        self.n_words_hover = config.plotting_parameters["n_words_hover"]
        self.plot_non_docs = plot_non_docs
        self.fig_title = self.get_fig_title()

    def plot(self):
        """
        Main function to handle plotting. Legend labels, hover labels are created,
        num docs classified to a topic, and the plot embeddings are created.

        Fig is created with visualize_documents from BERTopic, the fig is shown, and
        fig is saved to html if so specified.
        """
        print("Plotting documents...")

        if self.summarize_labels:
            words_legend = self.top_n_words(n_topics=self.num_topics, n_words=self.RAG_n_words_legend)
            legend_labels = get_summary_labels(words_legend, RAG=self.RAG)
            print(legend_labels)
            self.topic_model.set_topic_labels(legend_labels)
        else:
            words_legend = self.top_n_words(n_topics=self.num_topics, n_words=self.n_words_legend)
            legend_labels = self.make_legend_labels(words_legend)

        indices = get_sample_indices(self.topic_model, sample=self.sample)
        print("Number of docs sampled: ", len(indices))

        if self.summarize_docs:
            hover_labels = get_summary_sampled_docs(self.docs, indices, RAG=self.RAG)
        else:
            hover_labels = self.make_hover_labels()

        num_docs_topic = get_num_docs_topic(hover_labels)
        self.print_num_docs(num_docs_topic)
        plot_embeddings = self.make_plot_embeddings()

        fig = visualize_documents_(topic_model=self.topic_model,
                                   docs=hover_labels,
                                   indices=indices,
                                   reduced_embeddings=plot_embeddings,
                                   sample=self.sample,
                                   hide_document_hover=False,
                                   custom_labels=True,
                                   topics=list(range(self.n_total)),
                                   title=self.fig_title)
        fig.show()

        if self.save_html:
            file_name = self.get_param_str()
            fig.write_html(os.path.join(self.folder, file_name))

    def sample_docs_by_topic(self, sample_fraction=0.1):
        """
        Sample a given sample-size per topic, so if sample_size=0.1 10% of docs
        from each topic will be randomly sampled. Number of samples is floor-rounded
        to the nearest integer, ie 10% * 19 documents = 1 sample.

        Args:
            sample_fraction (float): sample size as fraction

        Returns:
            sampled_indices (lst[int]): sampled topic indices
        """
        np.random.seed(1)
        sampled_indices = []
        unique_topics = np.unique(self.topics)
        for t in unique_topics:
            indices = np.where(self.topics == t)[0]
            num_docs_for_topic = indices.shape[0]
            sampled_inds_topic = np.random.choice(indices, size=int(sample_fraction * num_docs_for_topic), replace=False)
            sampled_indices.extend(sampled_inds_topic)

        return sampled_indices

    def make_legend_labels(self, words_legend):
        """
        Function to make legend labels, which is the first n_words_legend words of each
        topic separated by ','. Set as labels via set_topic_labels() function within BERTopic.

        Returns:
            legend_labels (lst[str]): legend labels
        """
        print("Making legend labels...")
        legend_labels = [f"{i}: " + ", ".join(w) for i, w in enumerate(words_legend)]
        self.topic_model.set_topic_labels(legend_labels)  # legend_labels contains outlier topic label '0: the, and, of' but this is skipped automatically
        return legend_labels

    def top_n_words(self, n_topics=10, n_words=10):
        """
        Get top n_words words from the top n_topics topics

        Args:
            n_topics (int): number of topics to find words for.
            n_words (int): number of words to find for each topic.

        Returns:
            words_out (lst[lst[str]]): list of list of words
        """
        topic_words = self.topic_model.topic_representations_  # Words for each topic
        top_n_topics = {k: topic_words[k] for k in range(-1, n_topics)}  # Start at -1 because topic indexing starts at -1

        top_n_words_for_topics = []  # Out list will be 2d list with top n words for top m topics
        for values in top_n_topics.values():
            tuples = values[0:n_words]  # Tuples of words and probability
            top_n_words = [t[0] for t in tuples]  # Saving only the words
            top_n_words_for_topics.append(top_n_words)

        return top_n_words_for_topics

    def make_hover_labels(self):
        """
        Make hover labels for figure, which exists of a text part, so the first n_words_hover words
        of each topic separated by '|', and a size part, which is the topic size in percentage.
        For instance, hover label: "word1 | word2 | ... - Topic size: 4.25%"

        Returns:
            hover_labels (list[str]): hover labels
        """
        print("Making hoover labels...")
        # First, find text part of hover label
        words_hover = self.top_n_words(n_topics=self.num_topics, n_words=self.n_words_hover)
        hover_labels_text = [" | ".join(w) for w in words_hover]
        hover_labels_text = first_at_end(hover_labels_text)  # Add first element 'the|of|and..' to and of list; its topics id=-1, so the elements can be indexed by topic id

        # Second, assign the topic size to hoover labels
        counts = self.topic_model.topic_sizes_  # Count or size of all topics
        size_non_topic = counts[-1]  # The non-topic is indexed at -1
        total = self.num_docs - size_non_topic  # Total is the total number of classified documents, so subtract size of non-topic docs
        hover_labels_size = [words + f" - Topic size: {counts[i] / total * 100:.2f}%" for i, words in enumerate(hover_labels_text)]  # Combining words and topic size in single str

        # Setting hover_labels
        topics = self.get_topics()
        hover_labels = [hover_labels_size[t] if t != -1 else "" for t in topics]  # For index=-1, hover_label is "" because outlier docs are not plotted
        return hover_labels

    def make_plot_embeddings(self):
        """
        Make plot embeddings, so if self.plot_non_docs is False, filter out from the reduced embeddings
        the docs that belong to the outlier topic.

        Returns:
            plot_embeddings (np.array): 2d data points to plot
        """
        if not self.plot_non_docs:
            plot_embeddings = self.red_emb.copy()
            topics = self.get_topics()
            plot_embeddings[(topics == -1) | (topics >= self.n_total)] = 0  # Set embedding to 0 when id=-1 or id >= n_total
        else:
            plot_embeddings = self.red_emb
        return plot_embeddings

    def get_topics(self):
        """
        Get topics for self.topic_model, ie topic id for each of the texts in data

        Returns:
            np.array: topic ids for each of the text chunks, eg [-1, -1, 58, 17, 23, -1, ...]
        """
        return np.array(self.topic_model.topics_)

    def get_num_topics(self):
        """
        Get number of topics from self.topic_model

        Returns:
            num_topics (int): number of topics found
        """
        info = self.topic_model.get_topic_info()
        num_topics = info.shape[0] - 1
        print(f"Number of topics found: {num_topics}")
        return num_topics

    def print_num_docs(self, num_docs_topic):
        """
        Function to handle prints for number of documents

        Args:
            num_docs_topic (int): number of documents classified to a topic
        """
        if not self.plot_non_docs:
            print(f"Plotting only documents classified to a topic. Number of docs: {num_docs_topic}")
        else:
            print(f"Plotting all documents. Number of docs: {self.num_docs}")

    def get_param_str(self):
        return f"plot_mn{self.model_name}_n{self.n_total}_s{self.sample}.html"

    def get_fig_title(self):
        # OLD TITLE
        # self.fig_title = f"Text Data | Documents & Topics (merged)\n{self.model_name}" if self.merged else f"Text Data | Documents & Topics (unmerged)\n{self.model_name}"
        if self.merged:
            return f"(merged)\n{self.model_name}_sam{config.plotting_parameters['sample']}"
        else:
            return f"(unmerged)\n{self.model_name}_sam{config.plotting_parameters['sample']}"

    def get_sample_docs(self):
        pass
