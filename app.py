import streamlit as st
import os
import tempfile
import traceback
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IntelliDocs | Modern RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Light Mode)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #fdfdfe;
        background-image: radial-gradient(at 0% 0%, hsla(253,16%,7%,0.03) 0, transparent 50%), 
                          radial-gradient(at 50% 0%, hsla(225,39%,30%,0.03) 0, transparent 50%), 
                          radial-gradient(at 100% 0%, hsla(339,49%,30%,0.03) 0, transparent 50%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f1f5f9;
        padding-top: 1rem;
    }
    
    /* Title styling */
    .main-title {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        letter-spacing: -2px;
        margin-bottom: 0.1rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Message Bubbles */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }
    
    .stChatMessageContent {
        border-radius: 18px !important;
        padding: 1.2rem !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        max-width: 85%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    
    [data-testid="stChatMessageAssistant"] .stChatMessageContent {
        background: #ffffff !important;
        border: 1px solid #f1f5f9 !important;
        color: #1e293b !important;
    }
    
    [data-testid="stChatMessageUser"] .stChatMessageContent {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        margin-left: auto;
    }

    /* Input Box */
    .stChatInputContainer {
        border-radius: 15px !important;
        border: 1px solid #e2e8f0 !important;
        background: white !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05) !important;
        padding: 5px !important;
    }

    /* Primary Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: #0f172a;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 0;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: #1e293b;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
    }

    /* Glass Cards */
    .glass-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #f1f5f9;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
    }
    </style>
""", unsafe_allow_html=True)

def log_error(error_msg, doc_info="General"):
    """Saves detailed error logs to error_logs.txt"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n{'='*50}\n[{timestamp}] ERROR in {doc_info}\n{'-'*50}\n{error_msg}\n{traceback.format_exc()}\n{'='*50}\n"
    with open("error_logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

def process_documents(uploaded_files, api_key):
    try:
        documents = []
        for uploaded_file in uploaded_files:
            # Create a temporary file to load
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue
                
            documents.extend(loader.load())
            os.unlink(tmp_path) # Clean up
            
        if not documents:
            return None
            
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create Vector Store with Chroma
        import time
        time.sleep(1) # Small delay to stabilize connection
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vector_store = Chroma.from_documents(splits, embeddings)
        
        return vector_store
    except Exception as e:
        log_error(str(e), "Document Processing")
        st.error(f"Error processing documents: {str(e)}")
        return None

def get_rag_chain(vector_store, api_key):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        google_api_key=api_key,
        temperature=0.3
    )
    
    # 1. History-aware retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), contextualize_q_prompt
    )
    
    # 2. Main Q&A chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def main():
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("Configuration")
        
        google_api_key = st.text_input("Enter Google API Key", type="password")
        if not google_api_key:
            st.warning("Please enter your Google API Key to proceed.")
            st.info("You can get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
        
        st.divider()
        st.subheader("📚 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT)", 
            type=['pdf', 'txt'], 
            accept_multiple_files=True
        )
        
        if st.button("🚀 Index Documents") and uploaded_files and google_api_key:
            with st.spinner("Processing documents..."):
                st.session_state.vector_store = process_documents(uploaded_files, google_api_key)
                if st.session_state.vector_store:
                    st.success("Documents indexed successfully!")
                
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # Main Interface
    st.markdown('<h1 class="main-title">IntelliDocs</h1>', unsafe_allow_html=True)
    st.markdown("Your intelligent document companion. Upload files and start chatting!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        if google_api_key:
            if st.session_state.vector_store:
                try:
                    rag_chain = get_rag_chain(st.session_state.vector_store, google_api_key)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = rag_chain.invoke({
                                "input": prompt,
                                "chat_history": st.session_state.chat_history
                            })
                            
                            answer = response["answer"]
                            st.markdown(answer)
                            
                            # Display sources
                            with st.expander("🔍 View Sources"):
                                for i, doc in enumerate(response["context"]):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.caption(doc.page_content[:500] + "...")
                                    st.divider()
                            
                            # Update history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.session_state.chat_history.extend([
                                HumanMessage(content=prompt),
                                AIMessage(content=answer)
                            ])
                except Exception as e:
                    log_error(str(e), "Response Generation")
                    st.error(f"Error during response generation: {str(e)}")
            else:
                st.warning("Please upload and index documents first!")
        else:
            st.error("Please enter your Google API Key in the sidebar.")

if __name__ == "__main__":
    main()
