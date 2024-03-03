from typing import List

from scipy.cluster import hierarchy as sch

def find_unique_ids(lst):
    """
    Finds collections of ids that are longest and unique

    Args:
        lst (list[list[int]]): topic ids

    Returns:
        out_list (list[list[int]]): topic ids that are longest and unique.

    """
    lst.sort(key=len, reverse=True)
    ids = set()
    out_list = []
    for element in lst:
        if not any(i in ids for i in element):
            ids.update(element)
            out_list.append(element)
    return out_list

class Merge:

    def __init__(self, linkage_func):
        print("Merge object initialized")
        self.linkage_f = linkage_func  # single, complete, average, or ward

    def merge(self, topic_model, texts):
        pass

    def get_hierarchical_topics(self, topic_model, texts):
        linkage_function = lambda x: sch.linkage(x, self.linkage_f, optimal_ordering=True)
        hierarchical_topics = topic_model.hierarchical_topics(texts, linkage_function=linkage_function)
        return hierarchical_topics

    def get_topics_to_merge(self, hierarchical_topics):
        hierarch_topics_filtered = hierarchical_topics[hierarchical_topics["Distance"] < 0.5]
        topic_sets = hierarch_topics_filtered["Topics"].to_list()
        topics_to_merge = find_unique_ids(topic_sets)
        return topics_to_merge


