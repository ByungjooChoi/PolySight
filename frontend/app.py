import sys
import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Add project root to path to verify backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import backend modules
# We import inside a try-except block to handle path issues gracefully if needed, 
# but sys.path.append above should resolve it.
from backend.pipelines.ingestion import IngestionManager

# --- Setup ---
st.set_page_config(page_title="Elastic Visual Search", page_icon="🔍", layout="wide")
load_dotenv()

# Create temp dir if not exists (Safety check)
if not os.path.exists("temp"):
    os.makedirs("temp")

# Windows Asyncio Policy fix
# Only apply on Windows
if sys.platform.startswith('win'):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        # Some older python versions or environments might not have this
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
    token_ratio = st.slider("Token Pooling Ratio", 0.1, 1.0, 0.3, help="Lower ratio = smaller index size (Not yet fully wired to backend)")
    
    st.subheader("Text Settings")
    ocr_provider = st.selectbox("OCR Provider", ["jina", "reducto", "unstructured"])
    
    if st.button("Save & Apply Config"):
        # Update Environment Variables
        os.environ["ELASTIC_CLOUD_SERVERLESS_URL"] = elastic_url
        os.environ["ELASTIC_API_KEY"] = elastic_key
        os.environ["JINA_API_KEY"] = jina_key
        
        # Reset IngestionManager to reload engines with new keys
        IngestionManager._instance = None
        
        st.success("Configuration saved and applied!")

# --- Main Page ---
st.title("Elastic PDF Visual Search Comparison")
st.markdown("Upload a PDF to process it through **Visual Search (VLM)** and **Text Search (OCR)** pipelines.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file and st.button("Run Ingestion & Indexing"):
    file_path = os.path.join("temp", uploaded_file.name)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"Processing {uploaded_file.name}...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing pipelines...")
        progress_bar.progress(10)
        
        # Ensure IngestionManager is initialized (or re-initialized)
        # This will trigger model loading if not already loaded
        manager = IngestionManager()
        
        status_text.text("Running pipelines (this may take a while)...")
        progress_bar.progress(30)
        
        # Run async process
        # On Streamlit, we need to run the async loop carefully.
        # process_pdf is an async method.
        stats = asyncio.run(manager.process_pdf(file_path))
        
        progress_bar.progress(100)
        status_text.text("Processing Complete!")
        
        if stats:
            st.success("Ingestion Completed Successfully!")
            
            # Display Stats
            st.markdown("### Ingestion Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Visual Pages Indexed", stats.get("visual_count", 0))
            col2.metric("Text Chunks Indexed", stats.get("text_count", 0))
            col3.metric("Total Documents", stats.get("total_indexed", 0))
            
            st.json(stats) # Raw stats for debugging
            
        else:
            st.warning("Process completed but no data returned (Check logs).")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Optional: Cleanup temp file
        # os.remove(file_path)
        pass

# --- Footer ---
st.markdown("---")
st.caption("Powered by Elastic Cloud Serverless & Jina AI")
