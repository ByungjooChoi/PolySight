import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from backend.mcp_server.tools.comparison import compare_search_results

# Initialize Standard MCP Server
server = Server("elastic-visual-comparison")

# Register Tool Listing
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="compare_search_results",
            description="Compares search results between Visual Search (VLM) and Text Search (OCR).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "The user's search query to compare results for."
                    }
                },
                "required": ["query"]
            }
        )
    ]

# Register Tool Execution
@server.call_tool()
async def call_tool(name, arguments):
    if name == "compare_search_results":
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing 'query' argument")
            
        result = await compare_search_results(query)
        return [TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

# SSE Transport Setup
# This transport handles the SSE connection and message posting
sse = SseServerTransport("/messages")

async def handle_sse(request):
    """
    Handle the SSE connection.
    Elastic Agent Builder connects here to receive events.
    """
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        # Run the server with the read/write streams from the SSE connection
        await server.run(streams[0], streams[1], server.create_initialization_options())

async def handle_messages(request):
    """
    Handle incoming JSON-RPC messages via HTTP POST.
    """
    await sse.handle_post_message(request.scope, request.receive, request._send)

# Starlette App Routing
routes = [
    Route("/sse", endpoint=handle_sse),
    Route("/messages", endpoint=handle_messages, methods=["POST"])
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    # Run with Uvicorn
    # Accessible at http://localhost:8000/sse
    uvicorn.run(app, host="0.0.0.0", port=8000)
