from dbAPI import query_chromadb, queryBM25

def generate_response(
    prompt_input: str, collection=None, db_ready: bool = False, system_prompt: str = "",
    llm=None, BM25retriver=None, db_retrieval_amount: int = 3
) -> str:
    """
    Generate a response to a user's prompt using a combination of a language model and context from a database.

    Args:
        prompt_input: The user's input prompt.
        collection: The database collection to query for context.
        db_ready: Whether the database is ready to query.
        system_prompt: An optional system prompt to include in the input to the language model.
        llm: The language model to use.
        BM25retriver: An optional BM25 retriever to use for retrieving context.
        db_retrieval_amount: The number of documents to retrieve from the database.

    Returns:
        The generated response.
    """

    response = ""
    context = ""
    retrieved_docs = []

    if llm is None:
        return response

    if collection is not None and db_ready:
        retrieved_docs, _ = query_chromadb(prompt_input, collection=collection, n_results=db_retrieval_amount)
    if BM25retriver is not None:
        retrieved_docs.append(queryBM25(BM25retriver, prompt_input, db_retrieval_amount))
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            context += f"Reference document {i + 1}: {doc}\n"
    else:
        db_ready = False

    prompt = f"""
        System: {system_prompt}
        Question: {prompt_input}
        Context: {context}
        Answer:
        """
    # Create a chain: prompt -> LLM
    response = llm.invoke(prompt)
    return response


