import os
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings

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
    documents = []
    doc_ids = []
    all_splits = []

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
            loader = PyPDFLoader(file_path=os.path.join(SOURCE_PATH, file)) 
            if file not in docs:
                docs.append(file)
            async for page in loader.alazy_load():
                documents.append(page)
                for x in text_splitter.split_text(page.page_content):
                    if llmEmbedder is not None:
                        prompt = f"""
                            System: You are a helpful AI Embedder, your task is to review text split with the page content and provide addtional context for that text split.
                            Text Split: {x}
                            Page Content: {page.page_content}
                            Context:
                            """
                        AI_context = llmEmbedder.invoke(prompt)
                        x = AI_context + x
                    all_splits.append(x)
            doc_ids.append(file)

    return all_splits

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

