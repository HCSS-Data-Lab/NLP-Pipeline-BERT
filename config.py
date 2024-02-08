

parameters = {

    'def_clean_meth': 'def',  # Default text clean method
    'def_split_size': 'chunk',  # Default text split size
    'chunk_size': 500,  # Number of characters in chunk

    # Embedding and BERTopic parameters
    # 'bert_model': 'all-mpnet-base-v2',
    'bert_model': 'multi-qa-MiniLM-L6-cos-v1',  # The best small performer with large sequence length (seq len=512)
    'random_state': 0
}


