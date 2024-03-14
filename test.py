import re

import numpy as np
import math
import os
import pandas as pd

# from src.visualize_documents_func import visualize_documents_

from scipy.spatial import ConvexHull
import plotly.graph_objects as go

import matplotlib.pyplot as plt

def get_sample_indices(topic_per_doc, num_topics_in_fig=50, sample=1.0):
    np.random.seed(0)
    sample_indices = []
    print(set(topic_per_doc))
    for topic in list(set(topic_per_doc))[:num_topics_in_fig]:
        print(topic)
        indices_for_topic = np.where(np.array(topic_per_doc) == topic)[0]
        size = int(len(indices_for_topic) * sample)
        sample_indices.extend(np.random.choice(indices_for_topic, size=size, replace=False))
    indices = np.array(sample_indices)
    return indices



if __name__ == "__main__":

    topic_per_doc = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    num_topics = 3

    inds = get_sample_indices(topic_per_doc, num_topics, 0.5)
    print(inds)











