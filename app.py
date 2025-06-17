import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import tempfile
import time

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
try:
    from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
except ImportError:
    st.error("❌ Missing dependencies! Please install docx2txt by running: pip install docx2txt==0.8")
    st.stop()
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Additional PyTorch compatibility fix
import torch
torch.classes.__path__ = []

# Configure Streamlit page
st.set_page_config(
    page_title="DocuMind Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-header {
        padding: 1rem 0;
        text-align: center;
        background: linear-gradient(135deg, #1e2761 0%, #2d1b4e 100%);
        color: #fafafa;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #262730;
    }
    
    .section-box {
        padding: 1rem;
        background-color: #1a1d23;
        border-radius: 8px;
        border: 1px solid #2d3748;
        margin-bottom: 1rem;
    }
    
    .info-box {
        padding: 0.75rem;
        background-color: #1a202c;
        border-radius: 6px;
        border-left: 3px solid #4a90e2;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .warning-box {
        padding: 0.75rem;
        background-color: #2d1b1e;
        border-radius: 6px;
        border-left: 3px solid #f56565;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .success-box {
        padding: 0.75rem;
        background-color: #1a2e1a;
        border-radius: 6px;
        border-left: 3px solid #48bb78;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .sidebar-header {
        padding: 0.75rem;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #fafafa;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        border: 1px solid #4a5568;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: #fafafa;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        border: 1px solid #357abd;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #357abd 0%, #2c5aa0 100%);
        border-color: #2c5aa0;
    }
    
    .stButton > button:disabled {
        background: #2d3748;
        color: #718096;
        border-color: #2d3748;
    }
    
    /* Input styling */
    .stTextArea textarea {
        background-color: #1a202c;
        color: #fafafa;
        border: 1px solid #4a5568;
        border-radius: 6px;
    }
    
    .stSelectbox select {
        background-color: #1a202c;
        color: #fafafa;
        border: 1px solid #4a5568;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161b22;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a202c;
        color: #fafafa;
        border: 1px solid #4a5568;
        border-radius: 6px;
    }
    
    .streamlit-expanderContent {
        background-color: #1a202c;
        border: 1px solid #4a5568;
        border-top: none;
        border-radius: 0 0 6px 6px;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: #2d3748;
    }
    
    .stProgress .st-bp {
        background-color: #4a90e2;
    }
    
    /* Remove default margins */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Compact spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    h1, h2, h3 {
        color: #fafafa;
    }
    
    .stMarkdownContainer p {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

# Initialize session state
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'cross_encoder' not in st.session_state:
    st.session_state.cross_encoder = None

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded document file by converting it to text chunks.

    Takes an uploaded document file (PDF, DOCX, DOC), saves it temporarily, loads and splits 
    the content into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the document file

    Returns:
        A list of Document objects containing the chunked text from the document

    Raises:
        IOError: If there are issues reading/writing the temporary file
        ValueError: If the file type is not supported
    """
    # Get file extension
    file_extension = uploaded_file.name.lower().split('.')[-1]
    
    # Store uploaded file as a temp file with appropriate extension
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=f".{file_extension}", delete=False)
    temp_file_path = temp_file.name
    
    try:
        # Write the uploaded file content to temp file
        temp_file.write(uploaded_file.read())
        temp_file.flush()  # Ensure all data is written
        temp_file.close()  # Close the file handle before other processes use it
        
        # Load and process the document based on file type
        if file_extension == 'pdf':
            loader = PyMuPDFLoader(temp_file_path)
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        docs = loader.load()
        
        # Add file type to metadata
        for doc in docs:
            doc.metadata['file_type'] = file_extension
            doc.metadata['original_filename'] = uploaded_file.name
        
        # Split the documents into chunks (optimized for faster processing)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        return text_splitter.split_documents(docs)
        
    finally:
        # Ensure temp file is deleted even if an error occurs
        try:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except PermissionError:
            # On Windows, sometimes we need to wait a moment before deletion
            time.sleep(0.1)
            try:
                os.unlink(temp_file_path)
            except PermissionError:
                # If still can't delete, log a warning but don't crash
                st.warning(f"Could not delete temporary file: {temp_file_path}")
        except Exception as e:
            st.warning(f"Error cleaning up temporary file: {e}")


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Clears any existing documents in the collection first, then adds the new document splits.
    This ensures that only the current document is used for analysis.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues with the collection operations
    """
    collection = get_vector_collection()
    
    # Clear existing documents to ensure only current document is used
    try:
        # Get all existing IDs and delete them
        existing_data = collection.get()
        if existing_data['ids']:
            collection.delete(ids=existing_data['ids'])
    except Exception as e:
        # If collection is empty or doesn't exist, continue
        pass
    
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


def query_collection(prompt: str, n_results: int = 6):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 6.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. Optimized for faster response times.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    # Truncate context if too long to reduce processing time
    max_context_length = 1000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 300,  # Limit response length for faster generation
        },
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Provide concise, accurate answers based on the given context. Keep responses focused and direct.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


@st.cache_resource
def load_cross_encoder():
    """Load and cache the cross-encoder model for reuse."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def re_rank_cross_encoders(documents: list[str], query: str) -> tuple[str, list[int]]:
    """Re-ranks documents using a cached cross-encoder model for faster processing.

    Uses the cached MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 2 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.
        query: The search query for ranking relevance.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 2 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    # Use cached model for faster performance
    encoder_model = load_cross_encoder()
    ranks = encoder_model.rank(query, documents, top_k=2)  # Reduced from 3 to 2
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]] + " "
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text.strip(), relevant_text_ids


def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Main header
    st.markdown("""
    <div class="main-header">
        <h2>DocuMind Pro</h2>
        <p style="margin:0; color:#a0aec0;">Intelligent document analysis and Q&A platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Check Ollama connection
    if not check_ollama_connection():
        st.error("🚨 **Ollama not running!** Start Ollama service to continue.")
        st.markdown("""
        <div class="warning-box">
            <strong>Setup:</strong> Install Ollama → Pull models (llama3.2:3b, nomic-embed-text) → Restart app<br>
            <a href="https://ollama.ai" target="_blank">Get Ollama here</a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Sidebar for document upload
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h4>Document Upload</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>💡 Tips for best performance:</strong><br>
            • Supports PDF, DOCX, and DOC files<br>
            • Use text-based files (not scanned images)<br>
            • Upload multiple related documents for comprehensive analysis<br>
            • Keep total size reasonable for faster processing<br>
            • Ask specific, focused questions<br>
            • Each upload completely replaces previous documents
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "**Choose document files**",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            help="Select PDF, DOCX, or DOC files to upload and process"
        )

        # Reset document processed state if files are removed
        if not uploaded_files and st.session_state.document_processed:
            st.session_state.document_processed = False
            st.session_state.document_name = ""



        process = st.button(
            "🚀 Process",
            help="Process documents for Q&A",
            disabled=not uploaded_files or st.session_state.processing
        )

        if process and uploaded_files:
            st.session_state.processing = True
            
            with st.spinner("🔄 Processing documents..."):
                try:
                    # Show progress steps
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🧹 Clearing previous data...")
                    progress_bar.progress(5)
                    
                    all_documents_splits = []
                    total_files = len(uploaded_files)
                    
                    # Process each file
                    for idx, uploaded_file in enumerate(uploaded_files):
                        file_progress_start = 10 + (idx * 60 // total_files)
                        file_progress_end = 10 + ((idx + 1) * 60 // total_files)
                        
                        status_text.text(f"📖 Processing file {idx + 1}/{total_files}: {uploaded_file.name}")
                        progress_bar.progress(file_progress_start)
                        
                        # Process individual document
                        try:
                            file_splits = process_document(uploaded_file)
                        except ValueError as e:
                            st.error(f"❌ Unsupported file type for {uploaded_file.name}: {str(e)}")
                            continue
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                            continue
                        
                        # Add file identifier to metadata
                        for split in file_splits:
                            split.metadata['source_file'] = uploaded_file.name
                        
                        all_documents_splits.extend(file_splits)
                        progress_bar.progress(file_progress_end)
                    
                    # Check if any documents were successfully processed
                    if not all_documents_splits:
                        st.error("❌ No documents were successfully processed. Please check file formats and try again.")
                        st.session_state.document_processed = False
                        st.session_state.processing = False
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        status_text.text("✂️ Organizing all chunks...")
                        progress_bar.progress(75)
                        
                        status_text.text("🧠 Creating embeddings for all documents...")
                        progress_bar.progress(85)
                        
                        # Create combined filename for collection
                        combined_name = f"multi_doc_{len(uploaded_files)}_files"
                        add_to_vector_collection(all_documents_splits, combined_name)
                        
                        status_text.text("✅ All documents processed successfully!")
                        progress_bar.progress(100)
                        
                        st.session_state.document_processed = True
                        # Store file names as a comma-separated string
                        file_names = [f.name for f in uploaded_files]
                        st.session_state.document_name = f"{len(uploaded_files)} files: " + ", ".join(file_names[:3])
                        if len(uploaded_files) > 3:
                            st.session_state.document_name += f" and {len(uploaded_files) - 3} more"
                        
                        time.sleep(2)
                        progress_bar.empty()
                        status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.session_state.document_processed = False
                
                finally:
                    st.session_state.processing = False

        # Document status
        if st.session_state.document_processed:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Ready</strong> - Documents processed successfully
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⏳ No Documents</strong> - Upload and process documents first
            </div>
            """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="section-box">
            <h4>Ask Question</h4>
        </div>
        """, unsafe_allow_html=True)

        prompt = st.text_area(
            "**Question:**",
            placeholder="Enter your question about the document...",
            height=100,
            help="Be specific and concise for faster, better results"
        )

        ask = st.button(
            "🔍 Ask",
            disabled=not st.session_state.document_processed or not prompt.strip(),
            help="Get AI answer to your question"
        )

    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>How it works:</strong><br>
            Upload Documents → Process All → Ask Questions → Get AI Answers from All Files
        </div>
        """, unsafe_allow_html=True)

    # Handle question answering
    if ask and prompt and st.session_state.document_processed:
        st.markdown("---")
        st.markdown("### 🤖 AI Response")
        
        # Create progress tracking
        progress_placeholder = st.empty()
        
        try:
            # Step 1: Search documents
            progress_placeholder.info("🔍 Searching documents...")
            results = query_collection(prompt)
            context = results.get("documents")[0]
            
            if not context:
                progress_placeholder.empty()
                st.warning("⚠️ No relevant content found in the document for your question.")
            else:
                # Step 2: Re-rank results
                progress_placeholder.info("⚡ Ranking relevance...")
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context, prompt)
                
                # Step 3: Generate response
                progress_placeholder.info("🤖 Generating response...")
                progress_placeholder.empty()
                
                # Display the response
                st.markdown("**Answer:**")
                response_placeholder = st.empty()
                
                # Stream the response
                full_response = ""
                for chunk in call_llm(context=relevant_text, prompt=prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Show source information
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📚 Retrieved Chunks", expanded=False):
                        for i, doc in enumerate(context):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc[:150] + "..." if len(doc) > 150 else doc)
                            if i < len(context) - 1:
                                st.markdown("---")
                
                with col2:
                    with st.expander("🎯 Relevant Sections", expanded=False):
                        st.markdown(f"**Section IDs:** {relevant_text_ids}")
                        st.text(relevant_text[:300] + "..." if len(relevant_text) > 300 else relevant_text)
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.markdown("""
            <div class="warning-box">
                <strong>Check:</strong> Ollama running • Models installed • Rephrase question
            </div>
            """, unsafe_allow_html=True)

    elif ask and not st.session_state.document_processed:
        st.warning("⚠️ Upload and process documents first")
    
    elif ask and not prompt.strip():
        st.warning("⚠️ Enter a question first")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 1rem; font-size: 0.9rem;">
        <p>Powered by Ollama • ChromaDB • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
