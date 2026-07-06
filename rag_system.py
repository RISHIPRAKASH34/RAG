import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# -------------------------------
# Step 1: Load and chunk PDF
# -------------------------------
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# -------------------------------
# Step 2: Build FAISS index
# -------------------------------
def build_vector_db(chunks):
    if not chunks:
        raise ValueError("No chunks found. Check PDF path or text extraction.")

    # Hugging Face embeddings (stable and widely used)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -------------------------------
# Step 3: Query pipeline
# -------------------------------
def rag_query(vectorstore, query):
    # Embed query and retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Pass query + context to Ollama LLM
    llm = Ollama(model="llama3.2")  # You can swap with other Ollama models
    prompt = f"Answer the question based on context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return response

# -------------------------------
# Step 4: Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="ThinkAI RAG", layout="centered")
    st.title("ThinkAI RAG")

    # UI elements
    query = st.text_input("Enter your question:")
    run_button = st.button("Run", disabled=(query.strip() == ""))

    if run_button:
        with st.spinner("Embedding the query..."):
        # Load PDF and build vector DB once
            pdf_path = "Computer_Networking.pdf"  # Replace with your PDF file
            chunks = load_pdf(pdf_path)
            vectorstore = build_vector_db(chunks)
        
        with st.spinner("Generating response..."):
            answer = rag_query(vectorstore, query)
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()

