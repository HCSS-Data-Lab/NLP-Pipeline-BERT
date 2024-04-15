# BOOL variables whether to load data objects from file or not
LOAD_TEXT_SPLITS_FROM_FILE = False
LOAD_EMBEDDINGS_FROM_FILE = False
LOAD_REDUCED_EMBEDDINGS_FROM_FILE = False
LOAD_TOPIC_MODEL_FROM_FILE = False
LOAD_MODEL_EMBEDDINGS_FROM_FILE = True

clean_parameters = {
    'clean_text': False,  # Bool indicator whether to apply text cleaning
    'regex_ParlaMint': "\[\[|\]\]|ParlaMint.+?\s",  # Project-specific regex for ParlaMint data: remove brackets and datestamp starting with ParlaMint
}

dtm_parameters = {
    'keyword_find': 'None',  # None, search, tfidf

    'tfidf_threshold_type': 'value',  # value, document
    'tfidf_threshold': 0.8,

    'sample': False,
    'sample_size': 0.5
}

translate_param = {
    'translate': False,
    'target_lang': 'en',
    'source_lang': 'nl'
}

texts_parameters = {
    'clean_meth': 'vect',  # Text clean method: def (default, no cleaning), vect (vectorization), or ft (filter-texts function)
    'split_size': 'chunk_len',  # Text split size: chunk, chunk_len, sentence or sentence-pairs
    'chunk_size': 512,  # Number of characters in chunk
}

model_parameters = {
    # 'emb_model': 'all-MiniLM-L6-v2',  # A: Default model for BERTopic
    'emb_model': 'multi-qa-MiniLM-L6-cos-v1',  # B: The best small performer with large sequence length
    # 'emb_model': 'all-mpnet-base-v2',  # C: Current sentence-BERT state-of-the-art

    # 'emb_model': 'mixedbread-ai/mxbai-embed-large-v1',  # Best small performer from MTEB

    # 'emb_model': 'nl_core_news_sm',
    # 'spacy_exclude': ['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'],
}

bertopic_parameters = {
    'use_mmr': True,
    'mmr_diversity': 0.3,

    'use_pos': False,
    'spacy_mod_pos': 'en_core_web_sm',  # Spacy model used for Parts-of-Speech
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

countvectorizer_parameters = {  # CountVectorizer
    'ngram_range': (1, 3),
    'stop_words': 'english',
    # 'stop_words': ['de', 'het', 'een', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'of', 'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 'uit', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', 'u', 'want', 'nog', 'zal', 'me', 'zij', 'nu', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hun', 'dus', 'alles', 'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'kunnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'uw', 'iemand', 'geweest', 'andere'],
    'min_df': 0.01,
    'lowercase': False
}

kpcountvectorizer_parameters = {  # KeyphraseCountVectorizer
    'stop_words': 'english',
    'spacy_pipeline': 'en_core_web_sm',
}

umap_parameters = {
    'n_neighbors': 15,
    'n_components': 5,
    'min_dist': 0.0,
    'metric': 'cosine',
    'low_memory': False,
    'random_state': 0,
}

plotting_parameters = {
    'n_total': 50,   # Total number of topics to show in the fig
    'sample': 1,    # Sample (fraction) of docs to show in plot
    'n_words_legend': 3,    # Number of words to use in the description in the legend
    'n_words_hover': 6    # Number of words to display when hovering over figure
}

rag_parameters = {
    # RAG parameters
    'create_new_docs': False,
    'create_new_topics': False,
    'query_for_topic_labels': """Summarize these labels in one (sense-making) term that consists of
                            at MOST 4 words (captialize the first letter of every word, do NOT separate by comma):""",
    'query_docs_label': 'Summarize this texts in at MOST 3 terms and at most 5 words (captialize the first letter of every word and separate by comma):',
    'RAG_n_words_legend': 10,  # Number of noun frases to use to enhance the topic labels
    'LLM-model': "gpt-3.5-turbo",
    'temperature': 0.3,
    'article_retrievement': 10
}
