from llama_index.llms.openai import OpenAI
import os
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import TreeSummarize
import config
import asyncio

#OpenAI settings (key and model)
os.environ['OPENAI_API_KEY'] = config.rag_parameters["OPENAI_API_KEY"] #Should move to Gitignore (but since we are under closed HCSS repo, we keep it for now (ask James))
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3)

class RAG():
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
        
    def create_vector_store_index(self):
        """
        Create a vector store index based on embeddings and texts and save in appropriate folder.
        """
        nodes = [TextNode(text=self.texts[i], id_=str(i), embeddings=self.embeddings[i]) for i in range(0, len(self.texts))]
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
        if not self.RAG_from_file:
            self.create_vector_store_index()
        storage_context = StorageContext.from_defaults(persist_dir=self.path)
        index = load_index_from_storage(storage_context)
        
        print("Get New Labels from RAG") #Create query engine and deposit topics in query enginer
        query_engine = index.as_query_engine(similarity_top_k=10, llm=llm) #TODO move to config
        responses = []
        for topic_word in topics:
            response = query_engine.query(config.parameters["query_for_topic_labels"]+','.join(topic_word))
            responses.append(response.response)    
        return responses

    async def summarize_doc(self, docs):
        """
        Asynchronously loop over the selected docs to be summarized. Define method for summarization (Tree Summarize)
        
        TODO: qa_prompt can help enhance summarization prompting, find appropriate place
        Args:
            docs (List[str]): List of text documents to summarize
        Returns:
            responses (List[str]): the summarization of docs response text
        """
        #qa_prompt = PromptTemplate(config.parameters["query_docs_label"])
        summarizer = TreeSummarize(llm=llm, use_async=True)
        responses = []
        for doc in docs:
            response = await self.get_async_doc_response(summarizer, doc)
            responses.append(response)
        return responses

    async def get_async_doc_response(self, summarizer, doc):
        response = await summarizer.aget_response(config.parameters["query_docs_label"],doc)
        return response
            