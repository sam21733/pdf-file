import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# --- Step 1: Streamlit UI ---
st.set_page_config(page_title="Simple RAG App", layout="wide")
st.title("📄 Simple RAG with PDF + Streamlit")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # --- Step 2: Extract PDF text ---
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # --- Step 3: Split text into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # --- Step 4: Create embeddings & FAISS DB ---
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # --- Step 5: Set up LLM with retriever ---
    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        retriever=retriever
    )

    # --- Step 6: Chat UI ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about the PDF:")
    if query:
        result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((query, result["answer"]))

    # Display chat
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
