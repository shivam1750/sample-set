from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_pinecone import Pinecone
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]


def run_llm(query):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    index_name = "pinecone-chatbot"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=5000,
        temperature=0.45,
        token=HUGGINGFACEHUB_API_TOKEN,
    )
    chain = load_qa_chain(llm=llm, chain_type="stuff")

    docs = docsearch.similarity_search(query)
    out = chain.run(input_documents=docs, question=query)
    return out
