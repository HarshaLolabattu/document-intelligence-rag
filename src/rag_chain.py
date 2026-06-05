"""RAG chain: retrieve context + generate a grounded answer with OpenAI GPT."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_MODEL, LLM_TEMPERATURE


# Prompt that forces the model to answer ONLY from retrieved context.
# This is the "reduce hallucination" part: if the answer isn't in the
# context, the model is told to say so instead of making something up.
PROMPT_TEMPLATE = """You are a helpful assistant answering questions about documents.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever):
    """Build the full RAG chain: retrieve -> prompt -> GPT -> text answer."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def answer_question(chain, question: str):
    """Run a question through the RAG chain and return the answer."""
    return chain.invoke(question)
