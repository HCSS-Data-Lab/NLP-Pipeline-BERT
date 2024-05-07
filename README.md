# NLP Pipeline BERT

This repository does the following:
- Clean raw text bodies using regex
- Translate text using EasyNMT package
- Select documents relevant to keywords using TF-IDF
- Read cleaned text bodies from specified input folder
- Split text bodies in specified sizes (chunks variable in length; chunks with chunk_size number of character; sentences; or sentence pairs)
- Make embeddings from the split text parts
- Run topic modeling analysis with BERTopic module, which works in the following way:
    - Cluster text embeddings (high-dimensional vectors) with HDBSCAN clustering algorithm;
    - Extract topics from clusters with c-TF-IDF algorithm;
    - Reduce the dimensionality of the embeddings by mapping to a 2-dim space.
- Run dynamic topic modeling with BERTopic module:
    - Get timestamps for each of the text chunks
    - Find global topics over entire dataset
    - Find topics per timestamps
- Plotting the results
    - (Optional): Create a retrieval augmented generator
    - (Optional): Enhance topics labels from this generator
    - (Optional): Create doc labels from this generator
    - Topic modeling: plotting the documents, clustered as topics using (modified) default BERTopic visualize_documents function.
    - Dynamic topic modeling: plotting the topics over time, using (modified) BERTopic visualize_topics_over_time_ function.
    - Dynamic topic modeling: plotting % of documents relevant to keywords
- (Optional): Merging topic output to improve results
- (Optional): Evaluate topic output by calculating coherence

## Requirements

The `requirements.txt` is up to date.

## Running code
- Run script `main.py`, after specifying in the main:
  - `project_root`: which should be the folder `NLP-Pipeline-BERT`. The `project_root` should contain a folder `input` with a dataset name folder, for instance `ParlaMint` with, in turn, a folder `text_bodies` with input files in `.txt` format.
  - `dataset_name`: the dataset name, should correspond to a folder name in the `input` folder.
  - `task`: task can be `tm` (topic modeling) or `dtm` (dynamic topic modeling).
  - `years`: years should be a list of str years, like `[''2015']`. If `task=tm`, years can contain only a single year.
- Specify parameters in `config.py`:
  - `clean_parameters`: set `clean_text` to `True` and define a regex in `regex_[dataset_name]` to use for cleaning.
  - `translate_param`: set `translate` to `True` to translate texts from `source_lang` to `target_lang`.
  - `doc_selection_parameters`: 
      - `use_keyword_doc_selection`: set to `True` to use the keyword document selection. 
      - `select_dcouments`: set to `True` to select documents at runtime; if `False`, selected documents from an earlier run are used.
      - `doc_selection_method`: select relevant docs using TF-IDF (`tfidf`), select relevant docs using keyword search (`search`), sample random documents (`sample`).
      - `tfidf_threshold_type`: set number of documents based on cumulative TF-IDF value (`value`) or fraction of most relevant documents (`document`).
      - `tfidf_threshold`: threshold for TF-IDF, if `type=value` it is cumulative TF-IDF value; if `type=document` it is fraction of documents.
      - `sample_size`: sample size for random document sample.
  (The rest of the parameters are obvious or don't change)
- Initializing split texts, embeddings, reduced embeddings, and a trained topic model object at runtime takes several hours, depending on the dataset. To facilitate loading these data objects from a saved file, bool variables in `config.py` are specified and can be changed:
  - Loading split texts from file (LOAD_TEXT_SPLITS_FROM_FILE);
  - Loading text embeddings from file (LOAD_EMBEDDINGS_FROM_FILE);
  - Loading reduced embeddings from file (LOAD_REDUCED_EMBEDDINGS_FROM_FILE);
  - Loading a trained topic model object from file (LOAD_TOPIC_MODEL_FROM_FILE);
  - Using embeddings saved in file when training a new topic model object (LOAD_MODEL_EMBEDDINGS_FROM_FILE).
 
## Example usage

```commandline
python main.py
```

## devcontainer

Dev container is being build to ensure similarity between virtual environments.

## Folder structure

The project folder is structured as follows:

```text
├── .devcontainer
│
├── .github
│
├── .idea                                                   
│
├── input
    └── [project name]
          └── [year] 
              ├── raw_texts  # If project=ParlaMint, there must be a folder raw_texts; then text_bodies is created with TextCleaning class
              └── text_bodies  # Else, folder text_bodies must exist
│
├── output
    └── [project name]
        └── [year]
            ├── embeddings
            ├── figures
            ├── models
            └── texts  # This output folder structure is not required, if the folders do not exist they will be created at runtime
│                                         
├── src
    ├── analysis.py
    ├── embeddings_pp.py
    ├── evaluation.py
    ├── merge.py
    ├── plotting.py
    ├── init_files.py   
    ├── red_embeddings_pp.py  
    ├── translate.py
    └── texts_pp.py
│                                      
├── utils
    ├── visualize_documents_func.py    
    └── text_process_llm.py                             
│                                      
├── .gitignore
│
├── Dockerfile
│                                      
├── config.py
│                                      
├── get-pip.py
│
├── main.py
│                                      
└── requirements.txt
```