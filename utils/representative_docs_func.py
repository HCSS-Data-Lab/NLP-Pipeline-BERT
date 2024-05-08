import pandas as pd
from typing import List, Tuple, Union, Mapping, Any
from scipy.sparse import csr_matrix
import numpy as np
from bertopic.representation._mmr import mmr
from sklearn.metrics.pairwise import cosine_similarity


def _get_representative_docs_(topic_model, documents: pd.DataFrame, nr_repr_docs: int = 3):
    """ Get the nr_repr_docs most representative docs per topic

    Arguments:
        documents: Dataframe with documents and their corresponding IDs

    Returns:
        repr_docs: Populate each topic with nr representative docs
    """
    repr_docs, _, _, _ = _extract_representative_docs_(
        topic_model,
        topic_model.c_tf_idf_,
        documents,
        topic_model.topic_representations_,
        nr_samples=500,
        nr_repr_docs=nr_repr_docs
    )
    return repr_docs

def _extract_representative_docs_(topic_model,
                                  c_tf_idf: csr_matrix,
                                  documents: pd.DataFrame,
                                  topics: Mapping[str, List[Tuple[str, float]]],
                                  nr_samples: int = 500,
                                  nr_repr_docs: int = 5,
                                  diversity: float = None):  # -> Tuple[dict, List[Any], List[List[Union[int, Any]]], List[List[Any]]]:
    """
    #########################
    CUSTOM _extract_representative_docs function to have variable number of docs
    #########################


    Approximate most representative documents per topic by sampling
    a subset of the documents in each topic and calculating which are
    most represenative to their topic based on the cosine similarity between
    c-TF-IDF representations.

    Arguments:
        c_tf_idf: The topic c-TF-IDF representation
        documents: All input documents
        topics: The candidate topics as calculated with c-TF-IDF
        nr_samples: The number of candidate documents to extract per topic
        nr_repr_docs: The number of representative documents to extract per topic
        diversity: The diversity between the most representative documents.
                   If None, no MMR is used. Otherwise, accepts values between 0 and 1.

    Returns:
        repr_docs_mappings: A dictionary from topic to representative documents
        representative_docs: A flat list of representative documents
        repr_doc_indices: Ordered indices of representative documents
                          that belong to each topic
        repr_doc_ids: The indices of representative documents
                      that belong to each topic
    """
    # Sample documents per topic
    documents_per_topic = (
        documents.drop("Image", axis=1, errors="ignore")
        .groupby('Topic')
        .sample(n=nr_samples, replace=True, random_state=42)
        .drop_duplicates()
    )

    # Find and extract documents that are most similar to the topic
    repr_docs = []
    repr_docs_indices = []
    repr_docs_mappings = {}
    repr_docs_ids = []
    labels = sorted(list(topics.keys()))
    for index, topic in enumerate(labels):

        # Slice data
        selection = documents_per_topic.loc[documents_per_topic.Topic == topic, :]
        selected_docs = selection["Document"].values
        selected_docs_ids = selection.index.tolist()

        # Calculate similarity
        nr_docs = nr_repr_docs if len(selected_docs) > nr_repr_docs else len(selected_docs)
        bow = topic_model.vectorizer_model.transform(selected_docs)
        ctfidf = topic_model.ctfidf_model.transform(bow)
        sim_matrix = cosine_similarity(ctfidf, c_tf_idf[index])

        # Use MMR to find representative but diverse documents
        if diversity:
            docs = mmr(c_tf_idf[index], ctfidf, selected_docs, top_n=nr_docs, diversity=diversity)

        # Extract top n most representative documents
        else:
            indices = np.argpartition(sim_matrix.reshape(1, -1)[0], -nr_docs)[-nr_docs:]
            docs = [selected_docs[index] for index in indices]

        doc_ids = [selected_docs_ids[index] for index, doc in enumerate(selected_docs) if doc in docs]
        repr_docs_ids.append(doc_ids)
        repr_docs.extend(docs)
        repr_docs_indices.append([repr_docs_indices[-1][-1] + i + 1 if index != 0 else i for i in range(nr_docs)])
    repr_docs_mappings = {topic: repr_docs[i[0]:i[-1] + 1] for topic, i in zip(topics.keys(), repr_docs_indices)}

    return repr_docs_mappings, repr_docs, repr_docs_indices, repr_docs_ids
