import sys
import os

# Add project root to path to verify backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.utils.elastic_client import ElasticClient
    print("Successfully imported ElasticClient")
    
    client = ElasticClient()
    print("ElasticClient instantiated")
    
    # Check if client handles missing keys correctly (should be None or warning)
    if client.get_client() is None:
        print("ElasticClient correctly handled missing keys (client is None)")
    else:
        print("ElasticClient has a client object (Unexpected if keys are missing)")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
