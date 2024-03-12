# BOOL variables whether to load data objects from file or not
LOAD_TEXT_SPLITS_FROM_FILE = True
LOAD_EMBEDDINGS_FROM_FILE = True
LOAD_REDUCED_EMBEDDINGS_FROM_FILE = True
LOAD_TOPIC_MODEL_FROM_FILE = True
LOAD_MODEL_EMBEDDINGS_FROM_FILE = True

filter_parameters = {
    # Text filter parameters, used when reading text bodies
    # If True, a filter pattern should also be defined.
    'filter_Politie': False,

    'filter_ParlaMint': True,
    'filter_pattern_ParlaMint': "\[\[|\]\]|ParlaMint.+?\s",  # Clean regex for ParlaMint data: remove brackets and datestamp starting with ParlaMint
}

texts_parameters = {
    'clean_meth': 'vect',  # Text clean method: def (default, no cleaning), vect (vectorization), or ft (filter-texts function)
    'split_size': 'chunk',  # Text split size: chunk, sentence or sentence-pairs
    'chunk_size': 500,  # Number of characters in chunk
}

model_parameters = {
    # 'bert_model': 'all-MiniLM-L6-v2',  # A: Default model for BERTopic
    'bert_model': 'multi-qa-MiniLM-L6-cos-v1',  # B: The best small performer with large sequence length
    # 'bert_model': 'all-mpnet-base-v2',  # C: Current sentence-BERT state-of-the-art
}

bertopic_parameters = {
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
}

plotting_parameters = {
    'n_total': 50,   # Total number of topics to show in the fig
    'sample': 1.0,    # Sample (fraction) of docs to show in plot
    'n_words_legend': 3,    # Number of words to use in the description in the legend
    'n_words_hover': 6    # Number of words to display when hovering over figure
}

rag_parameters = {
    # RAG parameters
    'OPENAI_API_KEY': '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', #HCSS open_AI key
    'create_new_docs': False,
    'create_new_topics': True,
    'query_for_topic_labels':'Summarize these topic_labels in at MOST 3 words (captialized, without comma):',
    'query_docs_label':'Summarize this texts in at MOST 3 terms and at most 5 words (captialize the first letter of every word and separate by comma):'
}