import streamlit as st
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
import os

# Set up OpenAI API Key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Streamlit UI
st.title("Legal Document Analyzer")
st.write("Upload your legal document (PDF, DOCX, or TXT) to analyze.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

# Document processing function
def process_document(uploaded_file):
    # Read the file content
    if uploaded_file.type == "text/plain":
        document = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        # Use PyPDF or similar for extracting text from PDF
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        document = ""
        for page in reader.pages:
            document += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(uploaded_file)
        document = "\n".join([para.text for para in doc.paragraphs])

    return document

# If a file is uploaded
if uploaded_file is not None:
    document = process_document(uploaded_file)
    st.write("Document loaded. Extracting key insights...")

    # Split document into chunks for analysis
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(document)

    # Create OpenAI embeddings and FAISS vector store for document search
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Create a simple LLM chain for question answering
    qa_chain = load_qa_chain(OpenAI(temperature=0))

    # Text box for input question
    question = st.text_input("Ask a question about the document:")

    if question:
        # Perform similarity search and retrieve the most relevant chunk
        relevant_docs = vectorstore.similarity_search(question, k=3)

        # Use QA chain to answer the question based on relevant docs
        answer = qa_chain.run(input_documents=relevant_docs, question=question)
        st.write(f"Answer: {answer}")
    
    # Show the text chunks for reference
    st.subheader("Document Chunks (for reference):")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        st.write(f"Chunk {i+1}: {chunk[:500]}...")

