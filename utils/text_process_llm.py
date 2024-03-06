from typing import List
import asyncio

def get_summary_sampled_docs(docs: List[str], indices: List[int], RAG):
    """
    TODO: Save the dict(zip(summarized_docs,indices)) as pickle
    Get all the responses of the docs limits by summarizing it with GPT3.5-turbo. Use the asyncio function to run concurrently.
    Args:
        docs (List[str]) 
        indices: (List[int])
        RAG: (RAG class object) 
    Returns:
        summarized_docs (List[str]): 
    """
    print('Getting doc labels from RAG asynchronously...')
    summarized_docs = asyncio.run(RAG.summarize_doc(docs))
    return summarized_docs

def get_summary_labels(words_legend: List[str], RAG):
    """
    TODO: Save the enhanced_words_legend as pickle
    Get all the responses of the words topics by enhancing it with GPT3.5-turbo based on the created RAG from embeddings. 
    Since the topic words are not that large, synchonous calls suffice.
    Args:
        words_legend (List[str])
        RAG: (RAG class object) 
    Returns:
        enhanced_words_legend (List[str]): 
    """
    print('Getting word labels from RAG...')
    
    return RAG.summarize_words(words_legend)

    
    


