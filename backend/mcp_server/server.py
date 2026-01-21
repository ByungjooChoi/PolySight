"""
PolySight MCP Server
Exposes Visual Agent (MaxSim) and Text Agent (BM25) search tools
for Kibana Agent Builder integration.

Based on elastic/mcp-server-elasticsearch
"""
import sys
import os
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Configure logging (must use stderr for MCP stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("polysight-mcp")

# Import MCP server
from mcp.server.fastmcp import FastMCP

# Import tools
from backend.mcp_server.tools.comparison import (
    compare_search_results,
    get_index_status,
    search_visual_only,
    search_text_only
)

# Initialize FastMCP Server
mcp = FastMCP("polysight-agent-battle")

# Register Tools
mcp.add_tool(compare_search_results)
mcp.add_tool(get_index_status)
mcp.add_tool(search_visual_only)
mcp.add_tool(search_text_only)

logger.info("Registered MCP tools: compare_search_results, get_index_status, search_visual_only, search_text_only")


# Entry point
if __name__ == "__main__":
    logger.info("Starting PolySight Agent Battle MCP Server...")
    logger.info(f"Project root: {project_root}")

    # Check environment
    elastic_url = os.getenv("ELASTIC_CLOUD_SERVERLESS_URL")
    if elastic_url:
        logger.info(f"Elastic URL configured: {elastic_url[:30]}...")
    else:
        logger.warning("ELASTIC_CLOUD_SERVERLESS_URL not set!")

    # Run MCP server (handles stdio loop)
    mcp.run()
