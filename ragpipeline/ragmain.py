from .dataloader import DocumentLoader
from .vectordb import EmbeddingModel
from .vectordb import VectorDatabase
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
import torch

from sentence_transformers import CrossEncoder

class HybridRAG:

    def __init__(self,pdf_path,embedding_model,chunk_size,chunk_overlap,top_k_dense,top_k_bm25,rerank_top_n,enable_web_search,websearch_query,websearch_max_results,alpha,persist_directory="./chroma_db"):

        self.loader = DocumentLoader(pdf_path,chunk_size,chunk_overlap,enable_web_search,websearch_query,websearch_max_results) 
        self.chunks = self.loader.split()
        self.reranker = CrossEncoder("Qwen/Qwen3-Reranker-4B", device="cuda",model_kwargs={"torch_dtype": torch.bfloat16})
        print("loaded reranker")

        self.bm25_retriever = BM25Retriever.from_documents(self.chunks)#it converts its own invertable index
        self.bm25_retriever.k = top_k_bm25
        embedding_model = EmbeddingModel(embedding_model)
        self.vector_db = VectorDatabase(embedding_model=embedding_model.embedding,persist_directory= persist_directory,collection_name="rag_collection",)
        #print(torch.cuda.memory_summary())
        self.vector_db.add_documents(self.chunks)#direclt it will convert into vectors and store
        retriever = self.vector_db.get_retriever(search_type="mmr",k=top_k_dense,)#mmr or  'similarity','similarity_score_threshold' reranking occueres in mmr too
        print(torch.cuda.memory_summary())
        self.hybrid_retriever = EnsembleRetriever(retrievers=[retriever, self.bm25_retriever],weights=[alpha, 1-alpha])#retiver combiner and gives union of both retived documents
        #Reciprocal Rank Fusion 
        # self.reranker = CohereRerank(model="rerank-v3.5",top_n=rerank_top_n)
        print("Created vector db succesfully")
        self.top_n = rerank_top_n


    def retrieve(self, retrieval_query, rerank_query):

        # Retrieve documents
        docs = self.hybrid_retriever.invoke(retrieval_query)
        if not docs:
            return []

        # Remove duplicates
        docs = list({doc.page_content: doc for doc in docs}.values())

        # Prepare text for reranker
        document_texts = [doc.page_content for doc in docs]

        # Rerank retrieved documents
        rankings = self.reranker.rank(query=rerank_query,documents=document_texts,)

        # Return Top-N
        return [docs[item["corpus_id"]]for item in rankings[:self.top_n]]

    def merge_documents(self, *doc_lists):

        unique_docs = {}
        for docs in doc_lists:
            for doc in docs:
                unique_docs[doc.page_content] = doc

        return list(unique_docs.values())

    def cleanup(self):
        self.vector_db.delete_collection()
        print("Vector database deleted successfully")







    
    