import streamlit as st

import os
import time
import asyncio
from uuid import uuid4

import ollama
from langchain_ollama import OllamaLLM
from dbAPI import loadDocuments, getClient, getCollection, SOURCE_PATH, DB_PATH

## Utility functions
def initApp():
    # Create the directories if it does not exist    
    if not os.path.isdir(SOURCE_PATH):
        os.mkdir(SOURCE_PATH)
    else:
        for file in os.listdir(SOURCE_PATH):
            os.remove(os.path.join(SOURCE_PATH, file))

    if not os.path.isdir(DB_PATH):
        os.mkdir(DB_PATH)

    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = None
    if 'chatReady' not in st.session_state:
        st.session_state.chatReady = False
    if 'dropDown_model_list' not in st.session_state:
        st.session_state.dropDown_model_list = []
    if 'dropDown_embeddingModel_list' not in st.session_state:
        st.session_state.dropDown_embeddingModel_list = []
    if 'loaded_model_list' not in st.session_state:
        st.session_state.loaded_model_list = []
    if 'llm' not in st.session_state:
        st.session_state.llm = None

    # Define a collection for the RAG workflow
    if 'embedding' not in st.session_state:
        st.session_state.embedding = None
    if 'context_model' not in st.session_state:
        st.session_state.context_model = ""
    if 'embeddingModel' not in st.session_state:
        st.session_state.embeddingModel = ""
    if 'ollama_embedding_model' not in st.session_state:
        st.session_state.ollama_embedding_model = None
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client = getClient()       
    if 'docs' not in st.session_state:
        st.session_state.docs = []
    if 'newMaxTokens' not in st.session_state:
        st.session_state.newMaxTokens = 128
    if 'CRAG_iterations' not in st.session_state:
        st.session_state.CRAG_iterations = 5
    if 'overlap' not in st.session_state:
        st.session_state.overlap = 200
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1000
    if 'database_ready' not in st.session_state:
        st.session_state.database_ready = False
    if 'contextWindow' not in st.session_state:
        st.session_state.contextWindow = 2048
    if 'db_ready' not in st.session_state:
        st.session_state.db_ready = False
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function to query the ChromaDB collection
def query_chromadb(query_text, n_results=3):
    if st.session_state.collection is not None:
        results = st.session_state.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results["documents"], results["metadatas"]
    else:
        return [], []

def deleteDB():
    # remove sources
    for file in os.listdir(SOURCE_PATH):
        os.remove(os.path.join(SOURCE_PATH, file))

    # reset docs
    st.session_state.docs = []

    if st.session_state.collection is not None:
        # Define a collection for the RAG workflow
        st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
            name="rag_collection_demo",
            metadata={"description": "A collection for RAG with Ollama - Demo1"},
            embedding_function=st.session_state.embedding  # Use the custom embedding function
        )
        st.session_state.chroma_client.delete_collection("rag_collection_demo")
        st.session_state.chroma_client.reset()
    else:
        st.error("No DataBase is currently running")

def updateDB():
    st.session_state.db_ready = False
    # Check if the embedding model is selected
    if st.session_state.embeddingModel == "":
        st.error("Please select an embedding model")
        return
    
    # Checking if new files were updated
    if os.listdir(SOURCE_PATH) == st.session_state.docs:
        st.error("No new sources found. Please upload new sources for DB update.")
        return    

    with st.spinner("# Rebuilding DB - Please hold"):    
        # Load the documents
        contextLLM = None
        if st.session_state.context_model != "" and st.session_state.context_model != None:
            contextLLM = OllamaLLM(
                model=st.session_state.context_model,
                temperature=0.5,
                # other params... https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html
            )
        all_splits = asyncio.run(loadDocuments(st.session_state.chunk_size,st.session_state.overlap,st.session_state.docs, contextLLM))

        st.session_state.collection = getCollection(st.session_state.embeddingModel, st.session_state.chroma_client, "rag_collection_demo", "A collection for RAG with Ollama - Demo1")

        uuids = [str(uuid4()) for _ in range(len(all_splits))]
        if len(st.session_state.docs) > 0:
            st.session_state.collection.add(documents=all_splits, ids=uuids)
            st.session_state.db_ready = True  
        else:
            st.error("No PDF files found in the source folder. Please upload PDF files to proceed.")
            st.session_state.db_ready = False  
    
# Function to add documents to the ChromaDB collection
def updateOllamaModel():
    ollama_models = ollama.list()
    for model in ollama_models['models']:
        if 'embedding' not in model['model'] and 'embed' not in model['model']:
            if model['model'] not in st.session_state.dropDown_model_list:
                st.session_state.dropDown_model_list.append(model['model'])

    for model in ollama_models['models']:
        if 'embedding' in model['model'] or 'embed' in model['model']:
            if model['model'] not in st.session_state.dropDown_embeddingModel_list:
                st.session_state.dropDown_embeddingModel_list.append(model['model'])

def updateLoadedOllamaModel():
    loaded_models = ollama.ps()

    loaded_model_list = []
    for model in loaded_models['models']:
        loaded_model_list.append(model['model'])

    st.session_state.loaded_model_list = loaded_model_list

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

def generate_response(prompt_input):
    response = ""
    context = ""
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(prompt_input)
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            context += f"Reference document {i + 1}: {doc}\n"
    else:
        st.session_state.db_ready = False  

    if st.session_state.system_prompt == "":
        prompt = f"""
            System: You are a helpful assistant. here to answer questions and provide context. Answer as best as you can
            User Input: {prompt_input}
            Answer:
            """
    else:
        prompt = f"""
            System: {st.session_state.system_prompt}
            Question: {prompt_input}
            Context: {context}
            Answer:
            """
    # Create a chain: prompt -> LLM 
    response = st.session_state.llm.invoke(prompt)
    return response   

def updateMainOllamaModel():
    print(st.session_state.ollama_model)
    if st.session_state.ollama_model != None:
        st.session_state.chatReady = True
        st.session_state.llm = OllamaLLM(
            model=st.session_state.ollama_model,
            temperature=st.session_state.temperature,
            num_predict=st.session_state.newMaxTokens,
            num_ctx=st.session_state.contextWindow,
            # other params... https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html
        )
    else:
        st.session_state.chatReady = False

def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)

### App initialization
initApp()
updateOllamaModel()

### Page layout
st.title("🦙🦜🔗 OllamaRAG 🔗🦜🦙")

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input(key="User_prompt"):
    if st.session_state.ollama_model == None:
        st.error("Please select an Ollama model to proceed.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.messages[-1]["content"]) 
            # st.markdown(response) 
            st.write_stream(stream_data(response))
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

### Sidebar layout
with st.sidebar:
    st.header("Ollama model selection")
    st.write("You can find the list of available models at https://ollama.com/library")
    st.session_state.ollama_model = st.selectbox(
        "which ollama model would you like to use?",
        st.session_state.dropDown_model_list,
        index=None,
        placeholder="Select model...",
        key="ollama_model_selection"
    )    
    updateMainOllamaModel()

    if len(st.session_state.loaded_model_list) > 0:
        updateLoadedOllamaModel()
        st.write(f"Current loaded model:{st.session_state.loaded_model_list}") 
    else:
        st.write(f"No model is currently running")

    ollama_model_pull = st.text_input(
        "which ollama model would you like to pull? 🧲",
        placeholder="Enter model name",
    )        

    if ollama_model_pull:
        ollama.pull(ollama_model_pull)
        updateOllamaModel()

    st.session_state.system_prompt = st.text_area(
        "Enter System Prompt 👇",
        value="You are a helpful assistant.",
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.01,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.contextWindow = st.text_input(
            "Select LLM context window size",
            value=2048,
        )
        st.button('Check loaded models', on_click=updateLoadedOllamaModel)
    with col2:
        st.session_state.newMaxTokens = st.text_input(
            "Select new max tokens size",
            value=2048,
        )
        st.button("Clear chat history", on_click=clear_chat_history)


    st.header("RAG setup")

    if st.session_state.db_ready:
        st.markdown(''':green[Embedding DB ready]''')
    else:
        st.markdown(''':red[Embedding DB not ready]''')

    st.session_state.embeddingModel = st.selectbox(
        "Choose ollama embedding model",
        st.session_state.dropDown_embeddingModel_list,
        index=None,
        placeholder="Select model...",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.chunk_size = st.text_input(
            "Select vectorDB chunk size",
            value='1000',
        )
        st.session_state.CRAG_iterations = st.text_input(
            "Select CRAG iterations size",
            value='5',
        )
    with col2:
        st.session_state.overlap = st.text_input(
            "Select vectorDB overlap size",
            value='200',
        )
        st.session_state.context_model = st.selectbox(
            "Select context generation model",
            st.session_state.dropDown_model_list,
            index=None,
            placeholder = "Select model...",
        )        

    uploaded_files = st.file_uploader("Upload an article", 
                                        accept_multiple_files=True,
                                        type=('pdf'),
                                        )
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(SOURCE_PATH, uploaded_file.name)
            with open(save_path, mode='wb') as w:
                w.write(uploaded_file.getvalue())
        st.session_state.database_ready = False

    col1, col2 = st.columns(2)
    with col1:
        st.button('Rebuild DB', disabled=(st.session_state.embeddingModel is None), on_click=updateDB) 
    with col2:
        st.button('Delete DB', on_click=deleteDB) 
