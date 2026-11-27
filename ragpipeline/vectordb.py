import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import chain

class EmbeddingModel:

    def __init__(self,model_name):

        self.embedding = HuggingFaceEmbeddings(model_name=model_name,encode_kwargs={"normalize_embeddings": True,"batch_size": 16},model_kwargs={"device": "cuda:0"})
        print("loaded embedding model")
        

class VectorDatabase:

    def __init__(self,embedding_model,persist_directory,collection_name,):

        persist_path = Path(persist_directory)

        # Delete the entire old ChromaDB directory
        # if persist_path.exists():
        #     shutil.rmtree(persist_path)
        #     print("Deleted old ChromaDB")

        self.vector_store = Chroma(collection_name=collection_name,embedding_function=embedding_model,persist_directory=persist_directory,)
        # self.vector_store.reset_collection()#instaed of appending very time, it will reset DB for every run

    def add_documents(self, documents):
        batch_size = 100

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)

    def similarity_search(self, query, k):
        return self.vector_store.similarity_search(query,k=k,)

    def get_retriever(self,search_type,k):


        if search_type == "mmr":
            return self.vector_store.as_retriever(search_type=search_type,search_kwargs={"k": k,"fetch_k": 20,"lambda_mult": 0.8,},) # First retrieve the top 20 similar documents,70% relevance, 30% diversity

        elif search_type == "similarity_score_threshold":
            return self.vector_store.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": k,"score_threshold": 0.7,},)

        elif search_type == "similarity":
            return self.vector_store.as_retriever(search_type="similarity",search_kwargs={"k": k,},)
    def delete_collection(self):
        self.vector_store.delete_collection()