from typing import List
import asyncio
import os
import pickle 
import config

def get_summary_sampled_docs(docs: List[str], indices: List[int], RAG):
    """
    Get all the responses of the docs limits by etiher reloading or summarizing it with GPT3.5-turbo. Use the asyncio function to run concurrently.
    Args:
        docs (List[str]) 
        indices: (List[int])
        RAG: (RAG class object) 
    Returns:
        summarized_docs (List[str]): 
    """
    labels_path = os.path.join(RAG.path, "labels")
    if os.path.exists(labels_path+'/doc_labels.pkl') and config.rag_parameters["create_new_docs"]==False:
        print(f'Reloading stored doc labels of {len(indices)} texts...')
        with open(labels_path+'/doc_labels.pkl', 'rb') as file:
            docs = pickle.load(file)
    else:
        print(f'Getting doc labels from RAG asynchronously of {len(indices)} texts...')
        summarized_docs = asyncio.run(RAG.summarize_doc([docs[i] for i in indices if i < len(docs)]))
        #Go back to original docs
        for idx, summary in zip(indices, summarized_docs):
            if idx < len(docs):  # Check to avoid IndexError
                docs[idx] = summary


    #Save the summarized docs
    with open(labels_path+'/doc_labels.pkl', 'wb') as file:
        pickle.dump(docs, file)

    return docs

def get_summary_labels(words_legend: List[str], RAG):
    """
    Get all the responses of the words topics by enhancing it with GPT3.5-turbo based on the created RAG from embeddings. 
    Since the topic words are not that large, synchronous calls suffice.
    Args:
        words_legend (List[str])
        RAG: (RAG class object) 
    Returns:
        enhanced_words_legend (List[str]): 
    """
    print('Getting topic labels from RAG...')
    labels_path = os.path.join(RAG.path, "labels")
    if os.path.exists(labels_path+'/topic_labels.pkl') and config.rag_parameters["create_new_topics"]==False:
        print(f'Reloading stored topic labels of {len(words_legend)} labels...')
        with open(labels_path+'/topic_labels.pkl', 'rb') as file:
            words_legend = pickle.load(file)
    else:
        print(f"Getting topic labels from RAG asynchronously of {len(words_legend[:config.plotting_parameters['n_total']])} texts...")
        words_legend = RAG.summarize_words(words_legend)
        #summarized_topics = [item.split(', ') for item in summarized_topics]
        #words_legend = [summarized_topics[i] if i < config.parameters['n_total'] else words_legend[i] for i in range(len(words_legend))]

    #Save the summarized docs
    with open(labels_path+'/topic_labels.pkl', 'wb') as file:
        pickle.dump(words_legend, file)

    return words_legend

    
    


