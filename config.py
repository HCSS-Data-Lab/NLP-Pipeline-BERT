

texts_pp_params = {
    # Text preprocessing parameters

    # Parameters below in config or define in main?
    'split_text_size': 'chunk',  # Text section size to use for embeddings. Options: chunk, sentence, sentence-pairs
    'chunk_size': 1000  # Number of characters in chunk
}

emb_pp_params = {
    # Embedding pre-processing parameters
    'bert_model': 'all-mpnet-base-v2'
}

bertopic_params = {
    # BERTopic module parameters
    'bert_model': 'all-mpnet-base-v2',
    'random_state': 0
}

