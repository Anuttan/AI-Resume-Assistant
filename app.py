import os
import pdfplumber
import streamlit as st
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.tools import tool
import time  # For adding delay and progress bars

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

# Constants
RESUME_PATH = "resume.pdf"

# Function to extract text from resume (PDF)
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Load resume text and initialize FAISS vector store
def load_resume():
    if os.path.exists(RESUME_PATH):
        resume_text = extract_text_from_pdf(RESUME_PATH)
        vector_store = FAISS.from_texts([resume_text], embedding_function, metadatas=[{"resume_id": "Resume"}])
        return resume_text, vector_store
    return None, None

# Load resume on startup
st.session_state["resume_text"], st.session_state["vector_store"] = load_resume()

# Function to initialize RetrievalQA (RAG)
def get_rag_chain():
    if st.session_state["vector_store"] is None:
        return None
    retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever,
        return_source_documents=True
    )

# AI agent tool for fetching resume insights
@tool
def fetch_resume_summary(query: str):
    """Fetches a summary of the resume for a given question."""
    rag_chain = get_rag_chain()
    if rag_chain:
        response_data = rag_chain.invoke(query)  
        return response_data["result"]  
    return "No resume data available."

# Initialize AI Agent
tools = [fetch_resume_summary]
agent = initialize_agent(
    tools, 
    ChatOpenAI(model="gpt-4"), 
    agent="zero-shot-react-description",
    verbose=True
)

# Streamlit UI Setup
st.set_page_config(page_title="🎓 AI Resume Assistant", layout="wide")
st.title("📄 AI Resume Assistant")

st.write("👋 Welcome! This assistant analyzes your **resume.pdf** and provides intelligent insights. Choose a tab below to get started!")

# Create Tabs in the correct order
tab1, tab2, tab3 = st.tabs(["📜 Resume Summary", "🔍 Resume Q&A", "🧠 AI Analysis"])

# Resume Summary Tab
with tab1:
    st.subheader("📜 Auto-Generated Resume Summary")
    if st.button("🔄 Generate Summary", key="summary_submit"):
        if st.session_state["resume_text"]:
            with st.spinner("⏳ Generating your resume summary..."):
                time.sleep(2)  # Simulate processing delay
                summary_prompt = f"Summarize this resume in 3-4 sentences:\n\n{st.session_state['resume_text']}"
                summary_response = ChatOpenAI(model="gpt-4").invoke(summary_prompt)
                summary_text = summary_response.content  # ✅ FIXED: Extracting text properly
                st.success("✅ Summary Generated Successfully!")
                st.write("📢 **AI Summary:**")
                st.write(summary_text)
        else:
            st.warning("⚠️ No resume data available. Please ensure 'resume.pdf' is present.")

# Resume Q&A Tab
with tab2:
    st.subheader("🔍 Ask a question about the resume")
    query = st.text_input("💬 Enter your question:", key="qa_query")
    if st.button("🎯 Submit Question", key="qa_submit"):
        rag_chain = get_rag_chain()
        if rag_chain:
            with st.spinner("🔍 Searching resume for answers..."):
                time.sleep(2)  # Simulate processing delay
                response_data = rag_chain.invoke(query)
                response = response_data["result"]
                st.success("✅ Answer Found!")
                st.write("📝 **Response:**")
                st.write(response)
        else:
            st.warning("⚠️ No resume data available.")

# AI Analysis Tab
with tab3:
    st.subheader("🧠 AI-Powered Resume Insights")
    agent_query = st.text_input("📊 Enter an analysis request (e.g., 'Summarize experience'):", key="agent_query")
    if st.button("🚀 Submit Analysis", key="agent_submit"):
        with st.spinner("🤖 AI is analyzing the resume..."):
            time.sleep(3)  # Simulate processing delay
            agent_response = agent.run(agent_query)
            st.success("✅ Analysis Complete!")
            st.write("🔎 **AI Insights:**")
            st.write(agent_response)

# Sidebar: Resume Download Button
st.sidebar.subheader("📄 Download Resume")
if os.path.exists(RESUME_PATH):
    with open(RESUME_PATH, "rb") as file:
        st.sidebar.download_button(label="📥 Download Resume", data=file, file_name="resume.pdf", mime="application/pdf")
else:
    st.sidebar.warning("⚠️ Resume file not found.")
