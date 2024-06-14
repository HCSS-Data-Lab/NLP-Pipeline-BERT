# BOOL variables whether to load data objects from file or not
LOAD_TEXT_SPLITS_FROM_FILE = False
LOAD_EMBEDDINGS_FROM_FILE = True
LOAD_REDUCED_EMBEDDINGS_FROM_FILE = True
LOAD_TOPIC_MODEL_FROM_FILE = False
LOAD_MODEL_EMBEDDINGS_FROM_FILE = True
LOAD_TOPICS_OVER_TIME_FROM_FILE = False

clean_parameters = {
    'clean_text': True,  # Bool indicator whether to apply text cleaning
    # 'pattern': "\[\[|\]\]|ParlaMint.+?\s",  # Regex pattern for ParlaMint data: remove brackets and datestamp starting with ParlaMint (USED for first iteration)
    'pattern': r'\[\[.*?\]\]'  # Regex pattern for ParlaMint data: all text between brackets is removed (USED for paper)
}

translate_parameters = {
    'translate': False,
    'target_lang': 'en',
    'source_lang': 'nl'
}

doc_selection_parameters = {
    'use_keyword_doc_selection': False,  # If True, use document selection based on keywords
    'select_documents': False,  # If True, run the document selection functions using method below; If False, read documents selected based on keywords

    'doc_selection_method': 'tfidf',  # search, tfidf, sample

    'tfidf_threshold_type': 'value',  # value, document
    'tfidf_threshold': 0.8,
    # If threshold type is 'value', the threshold is the fraction of total tf-idf value to include,
    # e.g. if threshold is 0.8, it selects the top documents that contain 80% of cumulative tf-idf or relevance value.

    # If threshold type is 'document', the threshold number is the fraction of top documents to include,
    # e.g. if threshold is 0.2, it selects the 20% most relevant documents, so those with highest tf-idf score, with varying total relevance value.

    'sample_size': 0.5  # Sample size is used for randomly sampling documents
}

text_splitting_parameters = {
    'split_size': 'sentence',  # Text split size: chunk, chunk_len, sentence or sentence-pairs
    'chunk_size': 512,  # Number of characters in chunk
}

model_parameters = {
    'non_st_model': False,  # Non Sentence-Transformer model
    # 'emb_model': 'all-MiniLM-L6-v2',  # A: Default model for BERTopic
    # 'emb_model': 'multi-qa-MiniLM-L6-cos-v1',  # B: The best small performer with large sequence length (current in pipeline)
    'emb_model': 'all-mpnet-base-v2',  # C: Current Sentence Transformer state-of-the-art
    # 'emb_model': 'multi-qa-mpnet-base-dot-v1',  # Best Sentence Transformer with large sequence length (512)

    # 'emb_model': 'BAAI/bge-m3',

    # 'emb_model': 'nl_core_news_sm',
    # 'spacy_exclude': ['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'],
}

# Topic modeling parameters
tm_parameters = {
    'clean_meth': 'vect',   # Text clean method: def (default, no cleaning), vect (vectorization), or ft (filter-texts function)
    'use_mmr': True,
    'use_pos': False,
    'use_keyphrase': False,
    'use_custom_stopwords': True,
    'use_ctfidf': False,
    'update_topics': False,
}

##################
# Separate parameter dictionaries for function call in tm.py
##################

mmr_parameters = {
    'diversity': 0.8,
}

pos_parameters = {
    'model': 'en_core_web_sm',  # Spacy model used for Parts-of-Speech
    'pos_patterns': [  # PartsOfSpeech patterns
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}],  # illicit funds
        [{'POS': 'NOUN'}, {'POS': 'NOUN'}],  # world bank
        [{'POS': 'NOUN'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}],  # amazon rainforest conservation
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}],  # international crisis group
        [{'POS': 'ADJ'}, {'POS': 'ADJ'}, {'POS': 'NOUN'}],  # international monetary fund / united arab emirates
        [{'POS': 'NOUN'}],
        [{'POS': 'ADJ'}]
    ],
}

kpcountvectorizer_parameters = {  # KeyphraseCountVectorizer
    'stop_words': 'english',
    'spacy_pipeline': 'en_core_web_sm',
}

stop_words_parameters = {
    'custom_stopwords': ["The", "It", "We", "This", "What", "Can"],
}

countvectorizer_parameters = {  # CountVectorizer
    'ngram_range': (1, 3),
    'min_df': 0.01,
    'lowercase': False,
}

umap_parameters = {
    'n_neighbors': 15,
    'n_components': 5,  # From documentation: The dimension of space to embed into
    'min_dist': 0.0,
    'metric': 'cosine',
    'low_memory': False,
    'random_state': 0,
}

ctfidf_parameters = {
    'bm25_weighting': True,
    'reduce_frequent_words': True,
}

tm_plotting_parameters = {
    'n_total': 50,   # Total number of topics to show in the fig
    'sample': 1,    # Sample (fraction) of docs to show in plot
    'n_words_legend': 3,    # Number of words to use in the description in the legend
    'n_words_hover': 10,    # Number of words to display when hovering over figure
    'save_html': True
}

dtm_plotting_parameters = {
    'top_n_topics': 50,
    'custom_labels': False,
    'normalize_frequency': True,
    # 'topics': [1, 4, 5, 6, 7, 9, 11, 13, 14, 15],  # Topics to show
    # 'topics_background': [1, 4, 5, 9, 11, 13],  # Topics background
    # 'background_alpha': 0.2,
    # 'color_legend_opaque': False
}

rag_parameters = {
    # RAG parameters
    'create_new_docs': False,
    'create_new_topics': False,
    'query_for_topic_labels': """Summarize these labels in one (sense-making) term that consists of
                            at MOST 4 words (captialize thre first letter of evey word, do NOT separate by comma):""",
    'query_docs_label': 'Summarize this texts in at MOST 3 terms and at most 5 words (captialize the first letter of every word and separate by comma):',
    'RAG_n_words_legend': 10,  # Number of noun frases to use to enhance the topic labels
    'LLM-model': "gpt-3.5-turbo",
    'temperature': 0.3,
    'article_retrievement': 10
}
