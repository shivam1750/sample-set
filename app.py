import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain_pinecone import Pinecone
from PyPDF2 import PdfReader
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from main import run_llm

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
index_name = "pinecone-chatbot"


def get_pdf(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):

    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, api_env=PINECONE_API_ENV)
    index_name = "pinecone-chatbot"
    docsearch = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name)


def main():
    st.markdown(
        "<h1 style = 'text-align:center;'>Ask Questions to your PDFs</h1>",
        unsafe_allow_html=True,
    )
    query = st.text_area(label="Your query here", placeholder="Type something...")
    button = st.button("Generate")
    if button:
        st.write(run_llm(query))

    with st.sidebar:
        file = st.file_uploader("Upload Your PDF Here")
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf(file)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
