import sys
import os
import asyncio
import uuid
import streamlit as st
from dotenv import load_dotenv

# Set Hugging Face Cache to project directory to avoid Windows path limit
# This must be done before importing transformers or backend modules that use it
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
hf_cache_dir = os.path.join(project_root, "hf_cache")
os.environ["HF_HOME"] = hf_cache_dir
if not os.path.exists(hf_cache_dir):
    try:
        os.makedirs(hf_cache_dir, exist_ok=True)
    except Exception:
        pass

# Add project root to path to verify backend imports
sys.path.append(project_root)

# Import backend modules
from backend.pipelines.ingestion import IngestionManager

# --- Setup ---
st.set_page_config(page_title="Elastic Visual Search", page_icon="🔍", layout="wide")
load_dotenv()

# Create temp dir if not exists (Safety check)
if not os.path.exists("temp"):
    os.makedirs("temp")

# Windows Asyncio Policy fix
if sys.platform.startswith('win'):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("API Credentials")
    st.caption("Update keys here to override .env settings temporarily.")
    
    elastic_url = st.text_input("Elastic URL", value=os.getenv("ELASTIC_CLOUD_SERVERLESS_URL", ""), type="password")
    elastic_key = st.text_input("Elastic API Key", value=os.getenv("ELASTIC_API_KEY", ""), type="password")
    jina_key = st.text_input("Jina API Key", value=os.getenv("JINA_API_KEY", ""), type="password")
    
    st.subheader("Visual Settings")
    token_ratio = st.slider("Token Pooling Ratio", 0.1, 1.0, 0.3, help="Lower ratio = smaller index size")
    
    st.subheader("Text Settings")
    ocr_provider = st.selectbox("OCR Provider", ["jina", "reducto", "unstructured"])
    
    if st.button("Save & Apply Config"):
        os.environ["ELASTIC_CLOUD_SERVERLESS_URL"] = elastic_url
        os.environ["ELASTIC_API_KEY"] = elastic_key
        os.environ["JINA_API_KEY"] = jina_key
        
        # Reset IngestionManager to reload engines with new keys
        IngestionManager._instance = None
        
        st.success("Configuration saved and applied!")

# --- Main Page ---
st.title("Elastic PDF Visual Search Comparison")
st.markdown("Upload PDFs to process them through **Visual Search (VLM)** and **Text Search (OCR)** pipelines.")

# Enable multiple files
uploaded_files = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Run Ingestion & Indexing"):
    # Ensure list
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    total_files = len(uploaded_files)
    st.info(f"Starting ingestion for {total_files} file(s)...")
    
    overall_progress = st.progress(0)
    status_text = st.empty()
    success_count = 0
    
    # Initialize Pipelines
    try:
        status_text.text("Initializing pipelines (loading models)...")
        manager = IngestionManager()
    except Exception as e:
        st.error(f"Failed to initialize pipelines: {e}")
        st.stop()

    # Process Loop
    for i, uploaded_file in enumerate(uploaded_files):
        current_idx = i + 1
        status_text.text(f"Processing file {current_idx}/{total_files}: {uploaded_file.name}...")
        
        try:
            # Generate safe filename
            file_ext = os.path.splitext(uploaded_file.name)[1]
            if not file_ext:
                file_ext = ".pdf"
            safe_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join("temp", safe_filename)
            
            # Save file locally
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run Pipeline
            # process_pdf is async
            stats = asyncio.run(manager.process_pdf(file_path))
            
            if stats:
                success_count += 1
                with st.expander(f"✅ {uploaded_file.name} (Visual: {stats.get('visual_count')}, Text: {stats.get('text_count')})", expanded=False):
                    st.json(stats)
            else:
                st.warning(f"⚠️ {uploaded_file.name} processed but returned no data.")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            # Cleanup temp file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

        # Update Progress
        overall_progress.progress(current_idx / total_files)

    # Final Status
    status_text.text("Ingestion Complete!")
    if success_count == total_files:
        st.success(f"All {total_files} files processed successfully!")
    else:
        st.warning(f"Completed with issues. {success_count}/{total_files} files successful.")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Elastic Cloud Serverless & Jina AI")
