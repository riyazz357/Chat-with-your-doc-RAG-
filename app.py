import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
# --- CHANGE 1: Import the new Ollama LLM class ---
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Chat with Your Document", layout="wide")
st.title("📄 Chat with Your Document")

# --- CORE RAG FUNCTIONS ---

def get_vector_store(text):
    """Splits text, creates embeddings, and builds a FAISS vector store."""
    if not text:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_rag_prompt():
    """Creates a structured prompt template for the LLM."""
    template = """
    You are an expert assistant. Your task is to answer the user's question based ONLY on the following provided context.
    If the answer is not found within the context, you must say: "I'm sorry, but the provided information does not contain the answer to that question."

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# --- CHANGE 2: Initialize a real LLM from Ollama ---
# This connects to the Llama3 model you are running locally.
llm = Ollama(model="llama3")

# --- CHANGE 3: Create the RAG chain ---
# This chain combines the prompt and the LLM.
# It will pass the user's question and the retrieved documents to the prompt.
prompt_template = get_rag_prompt()
rag_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)


# --- STREAMLIT UI ---

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                string_data = uploaded_file.read().decode("utf-8")
                st.session_state.vector_store = get_vector_store(string_data)
                st.success("Document processed and indexed successfully!")
        else:
            st.warning("Please upload a document first.")

st.header("Chat Interface")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if st.session_state.vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Retrieve relevant documents
                retriever = st.session_state.vector_store.as_retriever()
                relevant_docs = retriever.get_relevant_documents(user_question)
                
                # --- CHANGE 4: Use the RAG chain to generate the answer ---
                # The chain automatically handles formatting the prompt with the
                # retrieved documents and the user question.
                response = rag_chain.invoke({
                    "input_documents": relevant_docs,
                    "question": user_question
                })
                
                # The actual answer is in the 'output_text' key
                answer = response['output_text']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please upload and process a document before asking questions.")