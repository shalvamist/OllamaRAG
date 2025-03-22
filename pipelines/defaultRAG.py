import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from CommonUtils.rag_utils import query_chromadb, query_bm25

def format_chat_history(messages):
    """Format chat history into a single string for context."""
    formatted_history = []
    for msg in messages[:-1]:  # Exclude the latest message as it's the current query
        role_prefix = "Assistant: " if msg["role"] == "assistant" else "Human: "
        formatted_history.append(f"{role_prefix}{msg['content']}")
    return "\n".join(formatted_history)

def generate_response(
    prompt_input: str, collection=None, db_ready: bool = False, system_prompt: str = "",
    llm=None, BM25retriver=None, db_retrieval_amount: int = 3
):
    """
    Generate a streaming response to a user's prompt using a combination of a language model and context from a database.

    Args:
        prompt_input: The user's input prompt.
        collection: The database collection to query for context.
        db_ready: Whether the database is ready to query.
        system_prompt: An optional system prompt to include in the input to the language model.
        llm: The language model to use.
        BM25retriver: An optional BM25 retriever to use for retrieving context.
        db_retrieval_amount: The number of documents to retrieve from the database.

    Yields:
        Chunks of the generated response as they become available.
    """
    if llm is None:
        yield ""
        return

    # Get chat history from session state
    chat_history = st.session_state.messages[:-1]  # Exclude current query
    history_text = format_chat_history(chat_history) if chat_history else ""
    
    # Initialize the context with system prompt and chat history
    context_parts = []
    if system_prompt:
        context_parts.append(f"System: {system_prompt}\n")
    if history_text:
        context_parts.append(f"Previous conversation:\n{history_text}\n")
    
    # Add RAG context if available
    if db_ready and collection is not None:
        # Get documents from vector search
        retrieved_docs, _ = query_chromadb(prompt_input, collection=collection, n_results=db_retrieval_amount)
        
        # Get documents from BM25 if available
        if BM25retriver is not None:
            bm25_docs = query_bm25(BM25retriver, prompt_input, db_retrieval_amount)
            
            # Combine both results
            rag_context = "Relevant information from documents:\n"
            seen_content = set()
            
            # Add vector search results
            for i, doc in enumerate(retrieved_docs, 1):
                if doc not in seen_content:
                    rag_context += f"Document {i}: {doc}\n\n"
                    seen_content.add(doc)
            
            # Add BM25 results
            for i, doc in enumerate(bm25_docs, len(seen_content) + 1):
                if doc.page_content not in seen_content:
                    rag_context += f"Document {i}: {doc.page_content}\n\n"
                    seen_content.add(doc.page_content)
            
            context_parts.append(rag_context)
        else:
            # Use only vector search results
            rag_context = "Relevant information from documents:\n"
            for i, doc in enumerate(retrieved_docs, 1):
                rag_context += f"Document {i}: {doc}\n\n"
            context_parts.append(rag_context)
    
    # Create the final prompt
    prompt_template = """
        {context}
        Human: {query}
        Assistant: Let me help you with that. """

    # Combine all context parts and format the prompt
    full_context = "\n".join(context_parts)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"]
    )
    
    formatted_prompt = prompt.format(
        context=full_context,
        query=prompt_input
    )
    
    # Generate streaming response
    for chunk in llm.stream(formatted_prompt):
        yield chunk 