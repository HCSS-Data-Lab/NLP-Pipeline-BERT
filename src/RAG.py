from llama_index.llms.openai import OpenAI
import os
from llama_index.core.schema import IndexNode
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import config
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
llm = OpenAI(model=config.rag_parameters['LLM-model'], temperature=config.rag_parameters['temperature'])  #
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/" + config.rag_parameters['bert_model'])
SERVICE_CONTEXT = ServiceContext.from_defaults(embed_model=embed_model)


class RAG:
    def __init__(self, embeddings, texts, RAG_from_file, path):
        """
        Create a RAG from the embeddings and texts that summarizes docs and enhances topic labels.
        Parameters:
            embeddings (np.array): Data Embeddings
            texts (list[str]): Original texts
            RAG_from_file (bool): Create rag from file or not
            path (str): Path to store or retrieve RAG

        Attributes:
            create_vector_store (None): Create the vectorindex based on embeddings and save it
            summarize_labels (list[str]): Get the topic labels based on the topic words
            summarize_docs (list[str]): Get summary labels of the docs

        """
        self.embeddings = embeddings
        self.texts = texts
        self.RAG_from_file = RAG_from_file
        self.path = path

    def create_vector_store_index(self, topics):
        """
        Create a vector store index based on embeddings and texts and save in appropriate folder.
        """

        nodes = [IndexNode(text=self.texts[i], index_id=str(i), embedding=self.embeddings[i].tolist()) for i in
                 range(len(self.texts))]
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=self.path)

    def summarize_words(self, topics):
        """
        TODO: Put similarity_top_k and LLM in configuration
        Summarize the words corresponding to topics, based on the vector store index.
        Either load or create the RAG from embeddings. Then create query engine and loop over all topics to enhance the topic description.
        Args:
            topics (List[str]): List of topics consisting of strings of the topic words
        Returns:
            response (List[str]): the output response text
        """
        print("Initiate RAG")  # Create RAG from embeddings or load existing RAG
        start_time = time.time()
        if not self.RAG_from_file:
            self.create_vector_store_index(topics)
        storage_context = StorageContext.from_defaults(persist_dir=self.path)
        index = load_index_from_storage(storage_context, service_context=SERVICE_CONTEXT)
        print('Creating or initiating the RAG took in total:', time.time() - start_time)
        print("Get new labels from RAG")  # Create query engine and deposit topics in query enginer
        query_engine = index.as_query_engine(similarity_top_k=config.rag_parameters['article_retrievement'], llm=llm)
        responses = []
        for topic_word in topics:
            response = query_engine.query(config.rag_parameters["query_for_topic_labels"] + ':'.join(topic_word))
            responses.append(response.response)
        return responses

    async def summarize_doc(self, docs):
        """
        Asynchronously loop over the selected docs to be summarized. Define method for summarization (Tree Summarize)

        Args:
            docs (List[str]): List of text documents to summarize
        Returns:
            responses (List[str]): the summarization of docs response text
        """
        summarizer = TreeSummarize(llm=llm, use_async=True)
        responses = []
        for doc in docs:
            response = await self.get_async_doc_response(summarizer, doc)
            responses.append(response)
        return responses

    async def get_async_doc_response(self, summarizer, doc):
        response = await summarizer.aget_response(config.rag_parameters["query_docs_label"], doc)
        return response
            