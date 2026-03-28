import streamlit as st
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


api_key = st.secrets["openai"]["openai_api_key"]
client = OpenAI(api_key=api_key)

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


def answer_question(question, relevant_docs):
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about legal documents using only the "
                    "provided context. If the answer is not in the context, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer clearly and concisely."
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()

# If a file is uploaded
if uploaded_file is not None:
    document = process_document(uploaded_file)
    st.write("Document loaded. Extracting key insights...")

    # Split document into chunks for analysis
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(document)

    # Create OpenAI embeddings and FAISS vector store for document search
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Text box for input question
    question = st.text_input("Ask a question about the document:")

    if question:
        # Perform similarity search and retrieve the most relevant chunk
        relevant_docs = vectorstore.similarity_search(question, k=3)

        # Answer the question based on the retrieved chunks
        answer = answer_question(question, relevant_docs)
        st.write(f"Answer: {answer}")
    
    # Show the text chunks for reference
    st.subheader("Document Chunks (for reference):")
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        st.write(f"Chunk {i+1}: {chunk[:500]}...")

