# NLP Pipeline BERT

This repository does the following:
- Read cleaned text bodies from specified input folder
- Pre-process the text bodies in the following way: split them in specified sizes (chunks with chunk_size number of character; sentences; or sentence pairs)
- Make embeddings from the split text parts
- Running topic modeling analysis with BERTopic module, which works in the following way:
    - Cluster text embeddings (high-dimensional vectors) with HDBSCAN clustering algorithm;
    - Extract topics from clusters with c-TF-IDF algorithm;
    - Reduce the dimensionality of the embeddings by mapping to a 2-dim space.
- Plotting the results
    - (Optional): Create a retrieval augmented generator
    - (Optional): Enhance topics labels from this generator
    - (Optional): Create doc labels from this generator
    - Plotting the documents, clustered as topics using (modified) default BERTopic visualize_documents function.
- (Optional): Merging topic output to improve results
- (Optional): Evaluate topic output by calculating coherence 

## Requirements

The `requirements.txt` is up to date.

## Running code
- Run script `main.py`, after specifying in the main:
  - `project_root`: which should be the folder `NLP-Pipeline-BERT`. The `project_root` should contain a folder `input`, which should contain a folder with the project name, for instance `Politie`, which should contain a folder `text_bodies` with input files in `.txt` format.
  - `project`: the project name, should correspond to a project name in the `input` folder.
- Specify parameters in `config.py`.
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