# NLP Pipeline BERT

This repository does the following:
- Read cleaned text bodies from specified input folder
- Pre-process the text bodies in the following way: split them in specified sizes (chunks with chunk_size number of character; sentences; or sentence pairs)
- Make embeddings from the split text parts
- Reduce the dimensionality of the embeddings by mapping them to a 2-dim space
- Running topic modeling analysis with BERTopic module
- Plotting the results
- (Optional): Merging topic output to improve results
- (Optional): Evaluate topic output by calculating coherence 

## Requirements

The `requirements.txt` is up to date.

## Running code
Run script `main.py`.
Specify parameters in `config.py`.
Specify in `main.py` whether to load split texts, embeddings, reduced embeddings, topic-model object from file or generate at runtime.
 
## Example usage

```commandline
python main.py
```

## Folder structure

The project folder is structured as follows:

```text
├── .devcontainer
│
├── .github
│
├── .idea                                                   
│                                      
├── src
    ├── analysis.py
    ├── embeddings_pp.py
    ├── evaluation.py
    ├── merge.py
    ├── plotting.py
    ├── preprocess.py   
    ├── red_embeddings_pp.py                         
    └── texts_pp.py
│                                      
├── utils                            
    └── visualize_documents_func.py                                      
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
├── requirements.txt
│                                      
└── test.py
```
