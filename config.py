

parameters = {

    'clean_meth': 'vect',  # Text cleaning method: def (default), vect (vectorization), or ft (filter-texts)
    'split_size': 'chunk',  # Text split size: chunk, sentence or sentence-pairs
    'chunk_size': 500,  # Number of characters in chunk

    # Embedding and BERTopic parameters
    # 'bert_model': 'all-MiniLM-L6-v2',  # A: Default model for BERTopic
    'bert_model': 'multi-qa-MiniLM-L6-cos-v1',  # B: The best small performer with large sequence length
    # 'bert_model': 'all-mpnet-base-v2',  # C: Current sentence-BERT state-of-the-art
    'random_state': 0,
    'use_mmr': False

}


