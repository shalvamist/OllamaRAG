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
    """
    Asynchronously load documents from the specified source path, split them into chunks, 
    and optionally generate AI contexts for each chunk using an LLM embedder.

    Args:
        chunk_size (int): The size of each text chunk.
        overlap (int): The overlap size between chunks.
        docs (list): The list of document names that have been loaded.
        llmEmbedder: The language model embedder for generating contexts.

    Returns:
        tuple: A tuple containing all text splits and their corresponding AI contexts.
    """
    all_splits = []
    AI_contexts = []

    # Initialize the text splitter with the specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(overlap),
        length_function=len,
        is_separator_regex=False,
    )

    # Iterate over the files in the source path
    for file in os.listdir(SOURCE_PATH):
        if file not in docs:
            # Load the file if it hasn't been processed yet
            loader = PyPDFLoader(file_path=os.path.join(SOURCE_PATH, file))
            docs.append(file)
            # Asynchronously load pages from the PDF
            async for page in loader.alazy_load():
                # Split the page content into chunks
                for split in text_splitter.split_text(page.page_content):
                    if llmEmbedder is not None:
                        # Generate an AI context for each chunk using the LLM embedder
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
                    # Add the split to the list of all splits
                    all_splits.append(split)

    return all_splits, AI_contexts

def getClient() -> chromadb.PersistentClient:
    """
    Returns a client to interact with the ChromaDB.

    The client is set up to use persistent storage in the current directory,
    and is configured to use the custom embedding function defined earlier.

    Returns:
        A PersistentClient object.
    """
    return chromadb.PersistentClient(path=DB_PATH, settings=CHROMA_SETTINGS)

def getCollection(embeddingModel, chroma_client, collectionName, collectionDescription):
    """
    Gets or creates a ChromaDB collection using the specified embedding model.

    Args:
        embeddingModel (str): The name of the Ollama embedding model to use.
        chroma_client (chromadb.PersistentClient): The ChromaDB client.
        collectionName (str): The name of the collection.
        collectionDescription (str): A description of the collection.

    Returns:
        A chromadb.Collection object.
    """
    # Create the custom embedding function using Ollama
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
    """
    Query the ChromaDB collection for relevant documents.

    Args:
        query_text (str): The input query.
        n_results (int): The number of top results to return.
        collection (chromadb.Collection): The ChromaDB collection to query.

    Returns:
        list of str: The top matching documents.
        list of dict: The metadata for the top matching documents.
    """
    if collection is not None:
        # Query the collection for relevant documents
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        # Return the top matching documents and their metadata
        return results["documents"], results["metadatas"]
    else:
        # Return empty lists if no collection is provided
        return [], []
    
def getBM25retriver(docs=[]):
    """
    Initialize a BM25 retriever and index the provided documents.

    Args:
        docs (list of str): The documents to index for retrieval.

    Returns:
        bm25s.BM25: An instance of the BM25 retriever with the corpus indexed.
    """
    # Create a BM25 model instance with the provided documents
    retriever = bm25s.BM25(corpus=docs)

    # Tokenize and index the documents in the BM25 model
    retriever.index(bm25s.tokenize(docs))

    # Return the initialized and indexed BM25 retriever
    return retriever

def queryBM25(retriever, query_text, n_results=3):
    """
    Query the BM25 retriever with the input query and retrieve the top n results.

    Args:
        retriever (bm25s.BM25): The BM25 retriever instance.
        query_text (str): The input query.
        n_results (int): The number of top results to return.

    Returns:
        list of str: The top matching documents.
    """
    # Tokenize the input query
    query_tokens = bm25s.tokenize(query_text)
    # Retrieve the top matching documents and their scores
    results, scores = retriever.retrieve(query_tokens, k=n_results)
    # Return the top matching documents
    return results
    