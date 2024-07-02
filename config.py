# BOOL variables whether to load data objects from file or not
LOAD_TEXT_SPLITS_FROM_FILE = False
LOAD_EMBEDDINGS_FROM_FILE = False
LOAD_REDUCED_EMBEDDINGS_FROM_FILE = False
LOAD_TOPIC_MODEL_FROM_FILE = False
LOAD_MODEL_EMBEDDINGS_FROM_FILE = True
LOAD_RAG_FROM_FILE = False
GENAI_TOPIC_LABELS, LOAD_GENAI_TOPIC_LABELS = False, False #It's only when the first is True that the second can be true
GENAI_DOC_LABELS, LOAD_GENAI_DOC_LABELS = False, False #It's only when the first is True that the second can be true

clean_parameters = {
    'filter_test': False,
    'filter_SKC': False,
    'filter_Politie': False,

    'filter_ParlaMint': False,
    'pattern_ParlaMint': "\[\[|\]\]|ParlaMint.+?\s",  # Clean regex for ParlaMint data: remove brackets and datestamp starting with ParlaMint
}

texts_parameters = {
    'clean_meth': 'vect',  # Text clean method: def (default, no cleaning), vect (vectorization), or ft (filter-texts function)
    'split_size': 'tokenize',  # Text split size: chunk, chunk_len, sentence or sentence-pairs
    'chunk_size': 512,  # Number of characters in chunk
}

model_parameters = {
    # 'bert_model': 'all-MiniLM-L6-v2',  # A: Default model for BERTopic
    'bert_model': 'multi-qa-MiniLM-L6-cos-v1',  # B: The best small performer with large sequence length
    # 'bert_model': 'all-mpnet-base-v2',  # C: Current sentence-BERT state-of-the-art

    # 'bert_model': 'mixedbread-ai/mxbai-embed-large-v1',  # Best small performer from MTEB
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
    'nr_topics':51,
    'update_topics': False,
    'use_keyphrase': False,
}

plotting_parameters = {
    'n_total': 51,   # Total number of topics to show in the fig
    'sample': 1,    # Sample (fraction) of docs to show in plot
    'n_words_legend': 3,    # Number of words to use in the description in the legend
    'n_words_hover': 6    # Number of words to display when hovering over figure
}

rag_parameters = {
    # RAG parameters
    'create_new_docs': False,
    'create_new_topics': False,
    'query_for_topic_labels':"""Summarize these words and underlying text chunks in at MOST 4 WORDS IN TOTAL and shorter if possible! Capitalize the first letter of every word and do not separate by commas. Only use the retrieved text chunks for summarizing. In total AT MOST 4 words!""",
    'query_docs_label':'Summarize this texts in at MOST 3 terms and at most 5 words (captialize the first letter of every word and separate by comma):',
    'RAG_n_words_legend':10, # Number of noun frases to use to enhance the topic labels
    'LLM-model':"gpt-3.5-turbo", #"gpt-4",
    'temperature':0.1,
    'article_retrievement':15
}
