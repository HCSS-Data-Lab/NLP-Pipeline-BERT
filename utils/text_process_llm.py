from typing import List
from src.RAG import RAG

def get_summary_sampled_docs(docs: List[str], indices: List[int], RAG):
    """
    Get all the responses of the docs limits by summarizing it with RAG-GPT3.5 turbo
    Args:
        words_legend (List[str])
        RAG: (RAG class object) 
    Returns:
        words_legend (List[str]): 
    """
    print('Getting doc labels from RAG...')
    return RAG.summarize_doc(docs)

def get_summary_labels(words_legend: List[str], RAG):
    """
    Get all the responses of the words topics by enhancing it with RAG-GPT3.5 turbo based on the text
    Args:
        words_legend (List[str])
        RAG: (RAG class object) 
    Returns:
        words_legend (List[str]): 
    """
    print('Getting word labels from RAG...')
    return RAG.summarize_words(words_legend)

    
    


