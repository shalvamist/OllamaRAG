import os
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
import bm25s

# Setting the embedding DB
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    allow_reset=True,
)

## App static variablesq
SOURCE_PATH = "./sources/"
DB_PATH = "./chroma_db/"
    
class ChromaDBEmbeddingFunction:
    """
    Custom embedding function for ChromaDB using embeddings from Ollama.
    """
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        # Ensure the input is in a list format for processing
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

async def loadDocuments(chunk_size=1000, overlap=200, docs=[], llmEmbedder=None):
    all_splits = []
    AI_contexts = []

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=int(chunk_size),
        chunk_overlap=int(overlap),
        length_function=len,
        is_separator_regex=False,
    )
    for file in os.listdir(SOURCE_PATH):
        # print(f'File {file} is being loaded...')
        with open(os.path.join(SOURCE_PATH, file), 'r') as f:
            if file not in docs:
                loader = PyPDFLoader(file_path=os.path.join(SOURCE_PATH, file)) 
                docs.append(file)
                async for page in loader.alazy_load():
                    for split in text_splitter.split_text(page.page_content):
                        if llmEmbedder is not None:
                            prompt = f"""
                                System: 
                                You are a helpful AI Embedder, Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
                                Page Content: 
                                <Document>
                                {page.page_content}
                                </Document>
                                Here is the chunk we want to situate within the whole document - 
                                <chunk>
                                {split}                            
                                </chunk>
                                Context:
                                """
                            AI_contexts.append(llmEmbedder.invoke(prompt))
                        all_splits.append(split)
    return all_splits, AI_contexts

def getClient():
    return chromadb.PersistentClient(path=DB_PATH, settings=CHROMA_SETTINGS) 

def getCollection(embeddingModel, chroma_client, collectionName, collectionDescription):
    embedding = ChromaDBEmbeddingFunction(
        OllamaEmbeddings(
            model=embeddingModel,
            base_url="http://localhost:11434"  # Adjust the base URL as per your Ollama server configuration
        )
    )
    # Define a collection for the RAG workflow
    return chroma_client.get_or_create_collection(
        name=collectionName,
        metadata={"description": collectionDescription},
        embedding_function=embedding  # Use the custom embedding function
    )

def query_chromadb(query_text, n_results=3, collection=None):
    if collection is not None:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results["documents"], results["metadatas"]
    else:
        return [], []
    
def getBM25retriver(docs=[]):
    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25(corpus=docs)
    retriever.index(bm25s.tokenize(docs))
    return retriever

def queryBM25(retriever, query_text, n_results=3):
    results, scores = retriever.retrieve(bm25s.tokenize(query_text), k=n_results)
    return results
    