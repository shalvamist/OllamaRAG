from dbAPI import query_chromadb

def generate_response(prompt_input, collection=None, db_ready=False, system_prompt="", llm=None):
    response = ""
    context = ""

    if llm is None:
        return response
    # Step 1: Retrieve relevant documents from ChromaDB
    retrieved_docs, metadata = query_chromadb(prompt_input, collection=collection)
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            context += f"Reference document {i + 1}: {doc}\n"
    else:
        db_ready = False  

    if system_prompt == "":
        prompt = f"""
            System: You are a helpful assistant. here to answer questions and provide context. Answer as best as you can
            User Input: {prompt_input}
            Answer:
            """
    else:
        prompt = f"""
            System: {system_prompt}
            Question: {prompt_input}
            Context: {context}
            Answer:
            """
    # Create a chain: prompt -> LLM 
    response = llm.invoke(prompt)
    return response 
