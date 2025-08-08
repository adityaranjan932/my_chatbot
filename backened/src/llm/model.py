# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI


# from ..retriver.retrive_info import load_retriever



# def create_qa_chain():
#     """
#     Create a RetrievalQA chain using Gemini and a custom chat prompt.
#     """
#     retriever = load_retriever()
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

#     system_template = """
# You are a smart general-purpose summarizer. You can summarize URLs, documents, and answer general questions.

# Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}

# Instructions:
# - Summarize the key points in a clear, structured format.
# - Do NOT fabricate any facts; only use content from the document.
# - If the information is missing, say "Information is not available."
# - Reference line numbers, clause numbers, headings, or page numbers when possible.
# - Maintain a professional and calm tone.
# - Extract and simplify lists if present.
# - If you do not understand the query or need clarification, ask the user.
# - You are summarizing a technical document. Identify its purpose, key components, and functionality.

# Question: {question}
# Answer:"""

#     prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system_template),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ])

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",  # Options: "stuff", "map_reduce", "refine"
#         chain_type_kwargs={"prompt": prompt},
#         return_source_documents=True
#     )

#     return qa_chain

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

from ..retriver.retrive_info import load_retriever


def create_qa_chain():
    """
    Create a ConversationalRetrievalChain using Gemini and a custom chat prompt,
    with memory to keep track of conversation history.
    """
    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Setup memory to track conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # required key
        return_messages=True
    )

    system_template = """
You are a smart general-purpose summarizer. You can summarize URLs, documents, and answer general questions.

Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Instructions:
- Summarize the key points in a clear, structured format.
- Do NOT fabricate any facts; only use content from the document.
- If the information is missing, say "Information is not available."
- Reference line numbers, clause numbers, headings, or page numbers when possible.
- Maintain a professional and calm tone.
- Extract and simplify lists if present.
- If you do not understand the query or need clarification, ask the user.
- You are summarizing a technical document. Identify its purpose, key components, and functionality.

Question: {question}
Answer:
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    # Create the chain using ConversationalRetrievalChain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    return qa_chain
