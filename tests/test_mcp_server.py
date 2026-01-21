import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.mcp_server.server import mcp
    print(f"MCP Server '{mcp.name}' initialized successfully.")
    print("Successfully imported server module.")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
