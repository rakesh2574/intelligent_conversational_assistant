import os
import streamlit as st
import time
import shutil
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import hashlib
from streamlit_lottie import st_lottie
import requests
import random

# For page config, we need to call it before any other Streamlit command
st.set_page_config(
    page_title="Taxmen AI v5",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Global Directories Setup
# -------------------------------
DOCUMENTS_DIR = "documents"
VECTORSTORE_DIR = "vectorstore"
TEMP_DIR = "temp_uploads"  # New temporary directory for custom PDF uploads
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)  # Ensure temp directory exists


# Function to clear temp directory
def clear_temp_directory():
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# -------------------------------
# CSS & UI Enhancements
# -------------------------------
st.markdown("""
<style>
/* Dark mode only */
.stApp {
    background: linear-gradient(135deg, #212529, #343a40);
    color: #f8f9fa;
}

/* Animated Title */
.animated-title {
    animation: titleAnimation 3s infinite;
    font-size: 3.2em;
    text-align: center;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6f61, #6b5b95, #88b04b);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: gradient 6s linear infinite;
    margin-bottom: 20px;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Card Styling - dark mode only */
.card {
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    background-color: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color: #f8f9fa;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}

/* Button Styling */
.stButton > button {
    border-radius: 20px;
    background: linear-gradient(90deg, #6b5b95, #4a69bd);
    color: white;
    font-weight: bold;
    padding: 0.6em 2em;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4a69bd, #6b5b95);
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}

/* Logout button styling */
button[kind="secondary"] {
    background: linear-gradient(90deg, #d63384, #dc3545);
    color: white;
}

button[kind="secondary"]:hover {
    background: linear-gradient(90deg, #dc3545, #d63384);
}

/* Enhanced Answer Button */
.enhanced-button > button {
    background: linear-gradient(90deg, #ff6f61, #d63384);
}

.enhanced-button > button:hover {
    background: linear-gradient(90deg, #d63384, #ff6f61);
}

/* Spinner Animation */
.spinner svg circle {
    animation: spinner-dash 1.5s ease-in-out infinite;
}

@keyframes spinner-dash {
  0% { stroke-dasharray: 1,150; stroke-dashoffset:0; }
  50% { stroke-dasharray: 90,150; stroke-dashoffset:-35; }
  100% { stroke-dasharray: 90,150; stroke-dashoffset:-124; }
}

/* Footer Notice */
.footer-notice {
    position: fixed; 
    bottom: 20px; 
    right: 20px; 
    font-weight: bold; 
    background: linear-gradient(90deg, #d63384, #6b5b95);
    color: white;
    padding: 10px 15px;
    border-radius: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(214, 51, 132, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(214, 51, 132, 0); }
    100% { box-shadow: 0 0 0 0 rgba(214, 51, 132, 0); }
}

/* Chat history styling - dark mode */
.chat-bubble {
    border-radius: 20px;
    padding: 15px;
    margin: 10px 0;
    max-width: 100%;
    animation: fadeIn 0.5s ease-in-out;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.question-bubble {
    background-color: #6b5b95;
    color: white;
    text-align: right;
    margin-left: 20%;
    border-bottom-right-radius: 0;
}

.answer-bubble {
    background-color: rgba(240, 242, 245, 0.1);
    color: #f8f9fa;
    margin-right: 20%;
    border-bottom-left-radius: 0;
    border: 1px solid rgba(128, 128, 128, 0.2);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Text area styling - dark mode */
.stTextArea textarea {
    border-radius: 15px;
    border: 2px solid #6b5b95;
    padding: 10px;
    font-size: 16px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    background-color: rgba(255, 255, 255, 0.05);
    color: #f8f9fa;
}

.stTextArea textarea:focus {
    border-color: #ff6f61;
    box-shadow: 0 0 0 2px rgba(255, 111, 97, 0.2);
    background-color: rgba(255, 255, 255, 0.1);
}

/* Login form styling - dark mode */
.login-container {
    max-width: 500px;
    margin: 0 auto;
    padding: 30px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.6s ease-in-out;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(128, 128, 128, 0.2);
    color: #f8f9fa;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

.login-title {
    font-size: 2em;
    text-align: center;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #ff6f61, #6b5b95, #88b04b);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: gradient 6s linear infinite;
    font-weight: bold;
}

.login-input {
    margin-bottom: 15px;
}

.login-button {
    background: linear-gradient(90deg, #6b5b95, #4a69bd);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 20px;
    cursor: pointer;
    font-weight: bold;
    width: 100%;
    margin-top: 10px;
    transition: all 0.3s ease;
}

.login-button:hover {
    background: linear-gradient(90deg, #4a69bd, #6b5b95);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Knowledge Base Selector */
.kb-selector {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(128, 128, 128, 0.2);
}

/* Custom File Uploader */
.upload-section {
    background-color: rgba(107, 91, 149, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    border: 1px dashed rgba(128, 128, 128, 0.4);
    transition: all 0.3s ease;
}

.upload-section:hover {
    background-color: rgba(107, 91, 149, 0.3);
    border-color: rgba(128, 128, 128, 0.6);
}

/* KB Mode Indicator */
.kb-mode {
    font-size: 0.9em;
    padding: 5px 10px;
    border-radius: 15px;
    display: inline-block;
    margin-bottom: 10px;
}

.mode-standard {
    background-color: #4a69bd;
    color: white;
}

.mode-custom {
    background-color: #ff6f61;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Lottie Animation Loader
# -------------------------------
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


# -------------------------------
# Authentication Helper using a form
# -------------------------------
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        # Only show the login form if not authenticated
        st.markdown('<div class="login-container">', unsafe_allow_html=True)

        # Login animation
        lottie_login = load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_bdo5lezz.json')
        if lottie_login:
            st_lottie(lottie_login, speed=1, height=200, key="login_animation")

        st.markdown('<h2 class="login-title">Welcome to Taxmen AI</h2>', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            # Hardcoded credentials: Username=Taxmen, Password=Taxmenv5
            if username == "Taxmen" and password == "Taxmenv5":
                st.session_state.authenticated = True
                # Clear temp directory on login
                clear_temp_directory()
                st.rerun()  # Rerun to update UI after login
            else:
                st.error("Invalid username or password")

        st.markdown('</div>', unsafe_allow_html=True)

        # Only show footer if not authenticated
        st.markdown('<div class="footer-notice">🚧 Taxmen AI v5 - Experimental upgrades in progress 🚧</div>',
                    unsafe_allow_html=True)
        st.stop()  # Stop execution here if not authenticated

    return True


# -------------------------------
# Document Processing Functions
# -------------------------------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return [page.extract_text() for page in pdf_reader.pages if page.extract_text()]


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n")
    return splitter.split_text(text)


def compute_documents_hash(documents_dir):
    hash_object = hashlib.sha256()
    for filename in sorted(os.listdir(documents_dir)):
        if filename.lower().endswith('.pdf'):
            with open(os.path.join(documents_dir, filename), "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_object.update(chunk)
    return hash_object.hexdigest()


def create_vectorstore(docs, metadatas):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadatas)


def save_vectorstore(vectorstore, file_path):
    vectorstore.save_local(file_path)


def load_vectorstore(file_path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)


def process_documents():
    current_hash = compute_documents_hash(DOCUMENTS_DIR)
    vectorstore_file = os.path.join(VECTORSTORE_DIR, "vectorstore.index")
    hash_file = os.path.join(VECTORSTORE_DIR, "documents_hash.txt")

    if os.path.exists(vectorstore_file) and os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            return load_vectorstore(VECTORSTORE_DIR)

    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
    docs, metadatas = [], []
    for pdf_file in pdf_files:
        with open(os.path.join(DOCUMENTS_DIR, pdf_file), "rb") as f:
            pages_text = extract_text_from_pdf(f)
            for page_number, page_text in enumerate(pages_text):
                chunks = chunk_text(page_text)
                docs.extend(chunks)
                metadatas.extend([{"source": pdf_file, "page": page_number + 1}] * len(chunks))

    vectorstore = create_vectorstore(docs, metadatas)
    save_vectorstore(vectorstore, VECTORSTORE_DIR)
    with open(hash_file, "w") as f:
        f.write(current_hash)
    return vectorstore


# New function to process a single uploaded PDF
def process_uploaded_pdf(uploaded_file):
    # Save the uploaded file to temp directory
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF
    with open(file_path, "rb") as f:
        pages_text = extract_text_from_pdf(f)

    docs, metadatas = [], []
    for page_number, page_text in enumerate(pages_text):
        chunks = chunk_text(page_text)
        docs.extend(chunks)
        metadatas.extend([{"source": uploaded_file.name, "page": page_number + 1}] * len(chunks))

    # Create vectorstore
    if docs:
        vectorstore = create_vectorstore(docs, metadatas)
        return vectorstore, f"Successfully processed {uploaded_file.name} ({len(docs)} chunks created)"
    else:
        return None, "Could not extract any text from the uploaded PDF. Please ensure the PDF contains extractable text."


# Function to check file size (5 MB limit)
def check_file_size(uploaded_file):
    # Get file size in bytes
    file_size = uploaded_file.size
    # Convert to MB
    file_size_mb = file_size / (1024 * 1024)
    # Check if within limit
    return file_size_mb <= 5


# LLM Response Enhancer Function
# -------------------------------
def enhance_response(original_answer, query):
    enhancer_prompt = f"""
You are a highly professional chatting professional, engaging, and empathetic tax advisor with deep expertise.
Your task is to rewrite the following answer to a tax question in an elegant, friendly, and beautifully articulated manner.
Ensure that your response has the following characteristics:
1. Warm, reassuring, and easy to read while maintaining accuracy
2. Well-structured with appropriate bullet points, numbered lists, or sections where relevant
3. Includes appropriate formatting like bold or italic for key points
4. Adds a touch of personality and empathy to make the user feel supported
5. Uses clear, concise language that explains complex tax concepts in accessible terms
6. Provides context and explanations where helpful

Original Question:
{query}

Original Answer:
{original_answer}

Enhanced Response (use markdown for formatting):
"""
    # Increase temperature to 0.7 for more creative and flowing language.
    enhancer_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)
    enhanced_answer = enhancer_llm.invoke(enhancer_prompt).content
    return enhanced_answer


# Question guardrail to ensure tax-related queries
# -------------------------------
def is_tax_related(question):
    """Check if the query is tax-related."""
    tax_guardrail_prompt = f"""
You are a specialist in determining if questions are related to taxes, accounting, finance, or business compliance.
Your task is to determine if the following question is related to taxes, accounting, finance, business compliance, or financial regulations.
Return ONLY 'YES' if it is tax/finance/accounting/business related, or 'NO' if it's about any other topic.

Question: {question}

Answer (ONLY 'YES' or 'NO'):
"""
    classification_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    response = classification_llm.invoke(tax_guardrail_prompt).content.strip().upper()
    return response == "YES"


def main():
    # Check authentication first
    check_password()

    # -------------------------------
    # Streamlit State Initialization
    # -------------------------------
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'custom_vectorstore' not in st.session_state:
        st.session_state.custom_vectorstore = None
    if 'kb_selection' not in st.session_state:
        st.session_state.kb_selection = "Existing Knowledge Base"
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'loading_animation' not in st.session_state:
        st.session_state.loading_animation = load_lottieurl(
            'https://assets3.lottiefiles.com/packages/lf20_usmfx6bp.json')
    if 'uploaded_file_info' not in st.session_state:
        st.session_state.uploaded_file_info = None

    # Add logout button with temp directory cleanup
    if st.sidebar.button("Logout", key="logout_button", type="secondary"):
        clear_temp_directory()
        st.session_state.authenticated = False
        st.rerun()

    # -------------------------------
    # Enhanced Title with Animation
    # -------------------------------
    st.markdown('<h1 class="animated-title">✨ Taxmen AI v5 ✨</h1>', unsafe_allow_html=True)

    # Welcome animation
    welcome_animation = load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_5ngs2ksb.json')
    if welcome_animation and len(st.session_state.chat_history) == 0:
        st_lottie(welcome_animation, height=200, key="welcome")
        st.markdown("""
        <div class="card">
            <h3>👋 Welcome to Taxmen AI!</h3>
            <p>I'm your AI tax assistant, ready to help with your tax-related questions. 
            Just type your query below and I'll provide you with accurate, helpful information.</p>
            <p>New feature: You can now upload your own PDF document and ask me questions about it!</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Knowledge Base Selection
    # -------------------------------
    st.markdown('<div class="kb-selector">', unsafe_allow_html=True)
    kb_options = ["Existing Knowledge Base", "Customized Knowledge Base"]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.kb_selection = st.radio(
            "Select Knowledge Base:",
            options=kb_options,
            index=kb_options.index(st.session_state.kb_selection),
            horizontal=True
        )

    with col2:
        if st.session_state.kb_selection == "Existing Knowledge Base":
            st.markdown('<div class="kb-mode mode-standard">Standard Mode</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kb-mode mode-custom">Custom Document Mode</div>', unsafe_allow_html=True)

    # File Upload for Custom Knowledge Base
    if st.session_state.kb_selection == "Customized Knowledge Base":
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("Upload Your Document")
        st.write("Upload a PDF file (max 5 MB) to ask questions about its content.")

        uploaded_file = st.file_uploader("", type="pdf", key="pdf_uploader")

        if uploaded_file:
            if check_file_size(uploaded_file):
                if st.button("Process Document", key="process_pdf"):
                    with st.spinner("Processing your document..."):
                        # Show loading animation
                        if st.session_state.loading_animation:
                            st_lottie(st.session_state.loading_animation, height=150, key="upload_loading")

                        # Process uploaded PDF
                        st.session_state.custom_vectorstore, st.session_state.uploaded_file_info = process_uploaded_pdf(
                            uploaded_file)

                        if st.session_state.custom_vectorstore:
                            st.success(st.session_state.uploaded_file_info)
                            # Clear conversation chain to rebuild with new vectorstore
                            st.session_state.conversation_chain = None
                        else:
                            st.error(st.session_state.uploaded_file_info)
            else:
                st.error("File size exceeds the 5 MB limit. Please upload a smaller file.")

        # Display info about the processed file
        if st.session_state.uploaded_file_info and st.session_state.custom_vectorstore:
            st.info(f"📄 Current document: {st.session_state.uploaded_file_info}")

            if st.button("Clear Custom Document", key="clear_custom"):
                st.session_state.custom_vectorstore = None
                st.session_state.uploaded_file_info = None
                st.session_state.conversation_chain = None
                clear_temp_directory()
                st.success(
                    "Custom document cleared. You can upload a new document or switch back to the existing knowledge base.")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Sidebar: API Key and Chat History
    # -------------------------------
    with st.sidebar:
        st.header("Configuration")

        openai_api_key = st.text_input("Enter OpenAI API key:", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            st.warning("Please provide your OpenAI API key.")

        # Add a nice separator
        st.markdown('<hr style="height:2px;border:none;color:#333;background-color:#333;">', unsafe_allow_html=True)

        # Display chat history with improved styling
        st.header("Chat History")
        if len(st.session_state.chat_history) == 0:
            st.info("Your conversation history will appear here.")
        else:
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div class="chat-bubble question-bubble">
                    <strong>You:</strong> {q}
                </div>
                <div class="chat-bubble answer-bubble">
                    <strong>Taxmen AI:</strong> {a.split("**Need more help?")[0] if "**Need more help?" in a else a}
                </div>
                """, unsafe_allow_html=True)

                # Add a small separator between conversations
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown('<hr style="margin: 15px 0; border:none; height:1px; background-color: #eee;">',
                                unsafe_allow_html=True)

    # -------------------------------
    # Conversational Chain Setup
    # -------------------------------
    if openai_api_key:
        # Choose appropriate vectorstore based on selection
        if st.session_state.kb_selection == "Existing Knowledge Base":
            if st.session_state.vectorstore is None:
                with st.spinner("Building knowledge base..."):
                    # Show loading animation
                    if st.session_state.loading_animation:
                        st_lottie(st.session_state.loading_animation, height=200, key="loading")
                    st.session_state.vectorstore = process_documents()
            active_vectorstore = st.session_state.vectorstore
        else:  # Customized Knowledge Base
            active_vectorstore = st.session_state.custom_vectorstore

        if active_vectorstore:
            # Only rebuild the conversation chain if it doesn't exist or if KB selection changed
            if st.session_state.conversation_chain is None:
                custom_prompt = PromptTemplate(
                    template="""You are an intelligent assistant. Provide accurate, detailed, and well-structured answers referencing the context provided below.Understand there could be numbers in queries, always make sure exact answers are captured from doc. If unsure, reply: 'Information not available.'

Context:
{context}


Question:
{question}

Answer (use markdown formatting for structure and emphasis):""",
                    input_variables=["context", "question"]
                )

                llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)
                st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=active_vectorstore.as_retriever(search_kwargs={"k": 8}),
                    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
                    combine_docs_chain_kwargs={"prompt": custom_prompt}
                )

    # Standard suffix to be appended when a valid answer is returned (only for tax knowledge base)
    suffix = "\n\n**Need more help? We're here for you!** Our tax professionals are also available for personalized assistance. [Contact Us](https://example.com/contact) or [WhatsApp Us](https://wa.me/1234567890)"

    # -------------------------------
    # Query Input Card
    # -------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.session_state.kb_selection == "Customized Knowledge Base" and not st.session_state.custom_vectorstore:
        question_placeholder = "Upload and process a document first to ask questions about it."
        disabled = True
    else:
        question_placeholder = "e.g., What tax deductions am I eligible for as a small business owner?" if st.session_state.kb_selection == "Existing Knowledge Base" else "Ask anything about your uploaded document..."
        disabled = False

    question = st.text_area("Enter your question:", height=120, placeholder=question_placeholder, disabled=disabled)

    st.markdown('<div class="enhanced-button">', unsafe_allow_html=True)
    ask_button = st.button("Ask Taxmen AI", disabled=disabled)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------
    # Query and Response Handling
    # -------------------------------
    if ask_button and question.strip():
        if not openai_api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not st.session_state.conversation_chain:
            st.error("Knowledge base still initializing, please wait.")
        else:
            # Check if question is tax-related (only for existing KB)
            is_valid_question = True
            if st.session_state.kb_selection == "Existing Knowledge Base":
                is_valid_question = is_tax_related(question)

            if not is_valid_question:
                st.error(
                    "I'm sorry, I can only answer questions related to taxes, accounting, finance, or business compliance. Please ask a tax-related question.")
            else:
                with st.spinner("Creating a comprehensive answer for you..."):
                    # Show animated thinking gif
                    if st.session_state.loading_animation:
                        st_lottie(st.session_state.loading_animation, height=200, key="enhanced_loading")

                    # Show a random tax fact while processing (only for tax KB)
                    if st.session_state.kb_selection == "Existing Knowledge Base":
                        tax_facts = [
                            "Did you know? The U.S. tax code is over 74,000 pages long!",
                            "Fun fact: The first income tax was created in 1861 to help finance the Civil War.",
                            "Tax trivia: Only about 0.7% of individual tax returns are audited by the IRS.",
                            "Tax fact: The word 'tax' comes from the Latin 'taxare' meaning 'to assess'.",
                            "Did you know? In the U.S., Tax Day is typically April 15, unless it falls on a weekend or holiday.",
                            "Tax trivia: The IRS was founded in 1862, primarily to fund the Civil War.",
                            "Fun fact: Albert Einstein once said: 'The hardest thing in the world to understand is the income tax.'",
                            "Tax fact: The first U.S. income tax was just 3% on income over $800 in 1861.",
                            "Did you know? The U.S. tax system is 'pay-as-you-go,' which is why taxes are withheld from paychecks.",
                            "Tax trivia: The 1040 tax form was introduced in 1913 and was just one page long."
                        ]
                        st.info(random.choice(tax_facts))
                    else:
                        st.info("Analyzing your document and crafting an answer...")

                    # Append numerical intent for tax questions only
                    if st.session_state.kb_selection == "Existing Knowledge Base":
                        numerical_intent = " Also, please include any relevant numbers, thresholds, currency limits, in AED or specific figures related to this question."
                        modified_question = question + numerical_intent
                    else:
                        modified_question = question

                    # Pass the modified question to the RAG pipeline
                    response = st.session_state.conversation_chain.invoke({'question': modified_question})
                    raw_answer = response.get('answer', '').strip()

                    if not raw_answer or raw_answer == "Information not available.":
                        final_answer = "Sorry, this is beyond my current knowledge; I am continuously learning. Please reach out to our experts for personalized assistance."
                        st.error(final_answer)
                    else:
                        # Enhance response for tax questions, keep as is for custom documents
                        if st.session_state.kb_selection == "Existing Knowledge Base":
                            final_answer = enhance_response(raw_answer, question)
                            final_answer += suffix
                        else:
                            final_answer = raw_answer

                        # Display success animation
                        success_animation = load_lottieurl(
                            'https://assets9.lottiefiles.com/packages/lf20_touohxv0.json')
                        if success_animation:
                            st_lottie(success_animation, height=150, key="success")

                        # Display the enhanced answer in a fancy card
                        st.markdown('<div class="card" style="border-left: 5px solid #ff6f61;">',
                                    unsafe_allow_html=True)
                        st.markdown("### 🌟 Your Answer:")
                        st.markdown(final_answer, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.session_state.chat_history.append((question, final_answer))

    # -------------------------------
    # Footer Notice
    # -------------------------------
    st.markdown('<div class="footer-notice">🚧 Experimental upgrades in progress 🚧</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    # Clear temp directory on application start
    clear_temp_directory()
    main()