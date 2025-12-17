# Based on the official elastic/mcp-server-elasticsearch
# Extended to support Visual Search Comparison features.

import sys
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from backend.mcp_server.tools.comparison import compare_search_results

# 1. Load Environment Variables
load_dotenv()

# 2. Configure Logging (Must use stderr for MCP stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-server")

# 3. Initialize FastMCP Server
# "elastic-visual-comparison" is the server name
mcp = FastMCP("elastic-visual-comparison")

# 4. Register Tools
# We can register the function directly if it has proper type hints and docstrings
mcp.add_tool(compare_search_results)

# 5. Entry point
if __name__ == "__main__":
    logger.info("Starting Elastic Visual Comparison MCP Server...")
    # mcp.run() handles the stdio loop
    mcp.run()
