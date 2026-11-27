from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import WebBaseLoader
from .websearch import WebSearch
from pathlib import Path
from docling.chunking import HierarchicalChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter




class DocumentLoader:
    
    def __init__(self,pdf_path,chunk_size,chunk_overlap,enable_web_search,websearch_query,websearch_max_results):#the index will be continues without becoming zero for every document

        self.pdf_path = Path(pdf_path)
        self.enable_web_search= enable_web_search
        self.query=websearch_query
        self.web_search = WebSearch(websearch_max_results)
        self.chunker = HierarchicalChunker()
        self.converter = DocumentConverter()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,add_start_index=True)
    
    def load(self):

        documents = []
        pdf_chunks = []
        for file in self.pdf_path.iterdir():
            if not file.is_file():
                continue
            try:        
                if file.suffix.lower() == ".pdf":
                    result = self.converter.convert(file)
                    for chunk in self.chunker.chunk(result.document):
                            pdf_chunks.append(Document(page_content=chunk.text,metadata={
                                "source": str(file),
                                "filename": file.name,}))
                    print(f"Loaded: {file.name}")

                else:
                    result = self.converter.convert(file)
                    documents.append(Document(page_content=result.document.export_to_markdown(),metadata={"source": str(file),"filename": file.name}))
                    print(f"Loaded: {file.name}")

            except Exception as e:
                print(f"Skipping {file.name}: {e}")

            #no single extractor works on every website,sometimes docling does not works with javascript heavy websites
            #for web search we have duckduckgosearch and for data extraction we have webbaseloader and docling
        if self.enable_web_search:
            #for web search we are not using HierarchicalChunker
            results = self.web_search.search(self.query)
            for result in results:
                url = result["link"]

                try:
                    doc = self.converter.convert(url)
                    documents.append(Document(page_content=doc.document.export_to_markdown(),metadata={"source": url,"title": result["title"],"type": "web",},))
                    print(f"Loaded Web (Docling): {url}")
                except Exception:

                    try:
                        loader = WebBaseLoader(url,header_template={"User-Agent": "Mozilla/5.0"}, requests_kwargs={"timeout": 20},
                        raise_for_status=True,bs_get_text_kwargs={"separator": "\n","strip": True,},)
                        for doc in loader.load():#langchain provides loader.load function  to convert data into langchain documents
                            text = "\n".join(line.strip()for line in doc.page_content.splitlines()if line.strip())
                            doc.page_content = text
                            doc.metadata.update({"title": result["title"],"type": "web",})
                            documents.append(doc)
                        print(f"Loaded Web (WebBaseLoader): {url}") 
                    except Exception as e:
                        print(f"Skipping {url}: {e}")
        self.documents = documents
        self.pdf_chunks = pdf_chunks

        return documents, pdf_chunks

    
    def split(self):
        documents, pdf_chunks = self.load()
        chunks = self.splitter.split_documents(documents)
        chunks.extend(pdf_chunks)
        self.chunks = chunks
        return chunks
