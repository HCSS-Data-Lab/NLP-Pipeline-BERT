import re
import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode, BaseNode, IndexNode
import pickle
import time
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import config

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

SERVICE_CONTEXT = ServiceContext.from_defaults(embed_model=embed_model)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3) #

if __name__ == "__main__":
    project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd()) 
    path = os.path.join(project_root,"output","Politie","texts","texts_chunk500_vect.pkl") 
    path2 = os.path.join(project_root,"output","Politie","embeddings","embeddings_multi-qa-MiniLM-L6-cos-v1_chunk500_vect.pkl") 
    print('hi')
    with open(path, "rb") as file:
        docs = pickle.load(file)
    
    with open(path2, "rb") as file:
        docs2 = pickle.load(file)
    print(type(docs['texts'])) 
    print(type(docs2['embeddings']))
    print(len((docs['texts'])),len((docs2['embeddings'])))
    print(type(docs2['embeddings'][0]))
    tijd=time.time()
    nodes = [IndexNode(text=docs['texts'][i], index_id=str(i), embedding = docs2['embeddings'][i].tolist()) for i in range(0,100)]
    index = VectorStoreIndex(nodes)
    project_root = os.environ.get(r'C:\Github\NLP-Pipeline-BERT', os.getcwd()) 
    path = os.path.join(project_root,"output","Politie","RAG") 
    index.storage_context.persist(persist_dir=path)
    print(time.time()-tijd)
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context,  service_context=SERVICE_CONTEXT)
    query_engine = index.as_query_engine(similarity_top_k=10, llm=llm) #TODO move to config
    response = query_engine.query("Give me three words about organized crime!")
    print(response.response)
































































