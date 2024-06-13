import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Load the GROQ and OpenAI API KEY
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("MCQ Generator")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
"""
Generate {num_questions} multiple choice questions based on the provided context. Each question should have four options, one of which is the correct answer.
Please provide the most accurate response based on the context.
<context>
{context}
<context>
Questions and Answers:
"""
)

def vector_embedding(pdf_file_path):
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    temp_file_path = os.path.join("temp_dir", uploaded_file.name)
    
    # Ensure the temp directory exists
    if not os.path.exists("temp_dir"):
        os.makedirs("temp_dir")

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Generate Embeddings"):
        with st.spinner("Processing..."):
            vector_embedding(temp_file_path)
            st.success("Vector Store DB is ready")

num_questions = st.number_input("Enter the number of questions you want", min_value=1, step=1)

if uploaded_file and num_questions > 0 and st.session_state.get("vectors"):
    if st.button("Generate MCQs"):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        query = {"input": "Generate questions", "num_questions": num_questions}
        start = time.process_time()
        response = retrieval_chain.invoke(query)
        st.write(f"Response time: {time.process_time() - start} seconds")
        st.write(response['answer'])
