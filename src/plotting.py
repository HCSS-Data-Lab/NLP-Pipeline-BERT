import numpy as np

def first_at_end(lst):
    """Append the first element of list at the end"""
    lst.append(lst.pop(0))
    return lst

class Plotting:

    def __init__(self, topic_model, reduced_embeddings, model_name, num_docs, save_html=True, merged=False):
        self.save_html = save_html
        self.topic_model = topic_model
        self.red_emb = reduced_embeddings
        self.merged = merged
        self.model_name = model_name
        self.num_docs = num_docs

        self.topics = np.array(self.topic_model.topics_)  # Topic id for each document, [-1, -1, 58, 17, 23, -1, ...]
        info = topic_model.get_topic_info()
        self.num_topics = info.shape[0] - 1
        print(f"Number of topics found: {self.num_topics}")

        # Plotting parameters
        self.n_total = 50  # Total number of topics to show in the fig
        self.sample = 1
        self.n_words_legend = 3  # Number of words to use in the description in the legend
        self.n_words_hover = 6  # Number of words to display when hovering over figure
        self.plot_non_docs = False
        self.fig_title = f"Text Data | Documents & Topics (merged)\n{self.model_name}" if self.merged else f"Text Data | Documents & Topics (unmerged)\n{self.model_name}"

    def plot(self):
        print("Plotting documents...")
        legend_labels = self.make_legend_labels()
        hover_labels = self.make_hover_labels()
        num_docs_topic = self.get_num_docs_topic(hover_labels)

        plot_embeddings = self.make_plot_embeddings(num_docs_topic, self.num_docs)
        fig = self.topic_model.visualize_documents(hover_labels,
                                                   reduced_embeddings=plot_embeddings,  # Add pre-defined reduced embeddings is faster than automatic in BERTopic module
                                                   hide_document_hover=False,
                                                   custom_labels=True,
                                                   topics=range(self.n_total),
                                                   sample=self.sample,
                                                   title=self.fig_title)

        fig.show()

    def make_legend_labels(self):
        print("Making legend labels...")
        words_legend = self.top_n_words(n_topics=self.num_topics, n_words=self.n_words_legend)
        legend_labels = [f"{i}: " + ", ".join(w) for i, w in enumerate(words_legend)]
        self.topic_model.set_topic_labels(legend_labels)  # legend_labels contains '0: the, and, of' but this is skipped automatically
        return legend_labels

    def top_n_words(self, n_topics=10, n_words=10):
        """Get top n words from topics
        Outlier topic with [the, of, and, ...] still included, this is dealt with automatically in plotting
        """
        topic_words = self.topic_model.topic_representations_
        top_n_topics = {k: topic_words[k] for k in range(-1, n_topics)}  # From -1 because topic indexing starts at -1

        words_out = []  # 2-D out list with description words
        for values in top_n_topics.values():
            tuples = values[0:n_words]  # Tuples of words and probability
            words = [t[0] for t in tuples]  # Saving only the words
            words_out.append(words)
        return words_out

    def make_hover_labels(self):
        print("Making hoover labels...")
        # Hoover labels consist of text and topic size
        # First, assign the correct text parts
        words_hover = self.top_n_words(n_topics=self.num_topics, n_words=self.n_words_hover)
        hover_labels_text = [" | ".join(w) for w in words_hover]  # Text for custom hoover labels, top 6 words separated by |
        hover_labels_text = first_at_end(hover_labels_text)  # Add first element 'the|of|and..' to and of list; its topics id=-1, so the elements can be indexed by topic id

        # Assign the topic size to hoover labels
        counts = self.topic_model.topic_sizes_  # Count or size of all topics
        size_non_topic = counts[-1]  # The non-topic is indexed at -1
        total = self.num_docs - size_non_topic  # Total is the total number of classified documents, so subtract size of non-topic docs
        hover_labels_size = [words + f" - Topic size: {counts[i] / total * 100:.2f}%" for i, words in enumerate(hover_labels_text)]  # Combining words and topic size in single str

        # Setting hover_labels and finding docs classified to a topic
        topics = self.get_topics()
        hover_labels = [hover_labels_size[t] if t != -1 else "" for t in topics]  # Indexing the hover labels by topic id, excluding -1 because this is non topic
        return hover_labels

    def get_num_docs_topic(self, hover_labels):
        """
        Find number of docs (text pieces) classified to a topic; the rest is classified to the 'outlier' topic.

        Args:
            hover_labels:

        Returns:
            int: Number of docs

        """
        return len(hover_labels) - hover_labels.count("")

    def make_plot_embeddings(self, num_docs_topic, num_docs):
        if not self.plot_non_docs:
            plot_embeddings = self.red_emb.copy()
            topics = self.get_topics()
            plot_embeddings[(topics == -1) | (topics >= self.n_total)] = 0  # Set embedding to 0 when id=-1 or id >= n_total
            print(f"Plotting only documents classified to a topic. Number of docs: {num_docs_topic}")
        else:
            plot_embeddings = self.red_emb
            print(f"Plotting all documents. Number of docs: {num_docs}")
        return plot_embeddings

    def get_topics(self):
        # Topic id for each document, [-1, -1, 58, 17, 23, -1, ...]
        return np.array(self.topic_model.topics_)

