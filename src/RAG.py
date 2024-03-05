from llama_index.llms.openai import OpenAI
import os
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize
import config
import random

#OpenAI settings (key and model)
os.environ['OPENAI_API_KEY'] = config.parameters["OPENAI_API_KEY"] #Should move to Gitignore (but since we are under closed HCSS repo, we keep it for now (ask James))
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)

class RAG():
    def __init__(self, embeddings, texts, RAG_from_file, path):
        """
        Create a RAG from the embeddings and texts that can create labels and
        Parameters:
            embeddings (str): Data Embeddings
            texts (str): Original texts
            RAG_from_file (bool): Create rag from file or not
            path (str): Path to store or retrieve RAG

        Attributes:
            create_vector_store (str): Create the vectorindex based on embeddings and save it
            summarize_labels (str): Get the topic labels based on the topic words
            summarize_docs (str): Get the labels of the docs

        """
        self.embeddings = embeddings
        self.texts = texts
        self.RAG_from_file = RAG_from_file
        self.path=path
        
    def create_vector_store_index(self):
        """
        Create a vector store index based on embeddings and texts and save in appropriate folder.
        """
        nodes = [TextNode(text=self.texts[i], id_=str(i), embeddings = self.embeddings[i]) for i in range(0,len(self.texts))]
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=self.path)

    def summarize_words(self, topics):
        """
        Summarize the words corresponding to topics, based on the vector store index.
        Args:
            topics (List[str]): List of topics consisting of strings of the topic words
        Returns:
            response (List[str]): the output response text
        """
        print("Initiate RAG")
        if not self.RAG_from_file:
            self.create_vector_store_index(self.path)
        storage_context = StorageContext.from_defaults(persist_dir=self.path)
        index = load_index_from_storage(storage_context)
        
        print("Get New Labels from RAG")
        query_engine = index.as_query_engine(similarity_top_k=10, llm=llm)
        responses = []
        for topic_word in topics:
            response = query_engine.query(config.parameters["query_for_topic_labels"]+','.join(topic_word))
            responses.append(response.response)
        print(responses)
        return responses

    def summarize_doc(self, docs):
        """
        Summarize the docs
        Args:
            docs (List[str]): List of text documents to summarize
        Returns:
            response (List[str]): the output response text
        """
        qa_prompt = PromptTemplate(config.parameters["query_docs_label"])
        summarizer = TreeSummarize()
        responses = []
        for doc in random.sample(docs, 5):
            response = summarizer.get_response(config.parameters["query_docs_label"],doc)
            print(doc, 'response',response)
            responses.append(response)
        return responses

        