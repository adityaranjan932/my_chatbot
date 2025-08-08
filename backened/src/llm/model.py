from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from ..retriver.retrive_info import load_retriever

def create_qa_chain():
    """
    Create a QA chain using Gemini and a custom chat prompt.
    """
    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

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
Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    return rag_chain

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

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
        return_messages=True,
        output_key="answer"  # specify which output to store in memory
    )

    system_template = """
You are a professional document analyst and information assistant. You provide well-structured, comprehensive answers from document content.

Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

RESPONSE FORMATTING GUIDELINES:
1. Start with a clear main heading that summarizes the topic
2. Use hierarchical structure with main sections and subsections
3. For definitions, include source clause/section references in parentheses
4. Use numbered lists for sequential items or criteria
5. Use bullet points for features, types, or related items
6. Bold important terms, amounts, and key concepts using **text**
7. Create clear section breaks between different topics
8. For tabular data (costs, limits, specifications), present in structured format
9. Always reference source document sections when available

CONTENT REQUIREMENTS:
- Extract information exactly as presented in the source documents
- Do NOT fabricate any information not present in the context
- If specific information is missing, state "Information is not available"
- Maintain professional, clear language
- Include relevant clause numbers, section references, or page numbers
- Present information in logical, easy-to-follow structure

EXAMPLE STRUCTURE:
**Main Topic – Definition and Overview**

**Definition (Source Reference)**
[Clear definition from document]

**Key Components/Types**
1. First type/component
2. Second type/component
   - Sub-detail A
   - Sub-detail B

**Requirements/Criteria**
- **Requirement 1:** Details
- **Requirement 2:** Details

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
