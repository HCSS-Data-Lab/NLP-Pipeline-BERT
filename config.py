

parameters = {

    'clean_meth': 'vect',  # Text cleaning method: def (default), vect (vectorization), or ft (filter-texts)
    'split_size': 'chunk',  # Text split size: chunk, sentence or sentence-pairs
    'chunk_size': 500,  # Number of characters in chunk

    # Embedding and BERTopic parameters
    # 'bert_model': 'all-MiniLM-L6-v2',  # A: Default model for BERTopic
    'bert_model': 'multi-qa-MiniLM-L6-cos-v1',  # B: The best small performer with large sequence length
    # 'bert_model': 'all-mpnet-base-v2',  # C: Current sentence-BERT state-of-the-art

    'random_state': 0,

    'use_mmr': True,
    'mmr_diversity': 0.3,

    'use_pos': False,
    'spacy_mod_pos': 'en_core_web_sm',
    'pos_patterns': [  # PartsOfSpeech patterns
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}],  # illicit funds
        [{'POS': 'NOUN'}, {'POS': 'NOUN'}],  # world bank
        [{'POS': 'NOUN'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}],  # amazon rainforest conservation
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}],  # international crisis group
        [{'POS': 'ADJ'}, {'POS': 'ADJ'}, {'POS': 'NOUN'}],  # international monetary fund / united arab emirates
        [{'POS': 'NOUN'}],
        [{'POS': 'ADJ'}]
    ],

    'update_topics': False,
    'use_keyphrase': False,

    # Plotting parameters
    'n_total': 50,   # Total number of topics to show in the fig
    'sample': 0.1,    # Sample (fraction) of docs to show in plot
    'n_words_legend': 3,    # Number of words to use in the description in the legend
    'n_words_hover': 6    # Number of words to display when hovering over figure

}


