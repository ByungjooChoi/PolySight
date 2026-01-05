# Pipelines package
import os

# Set HF_HOME to local directory to avoid Windows path length issues
# This must be done before importing transformers in any pipeline module
# Calculate project root from backend/pipelines/__init__.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
hf_cache_dir = os.path.join(project_root, "hf_cache")

# Set environment variable
os.environ["HF_HOME"] = hf_cache_dir

# Ensure hf_cache directory exists
if not os.path.exists(hf_cache_dir):
    try:
        os.makedirs(hf_cache_dir, exist_ok=True)
    except Exception:
        pass
