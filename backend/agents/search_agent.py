"""
Search Agent - LLM-powered agent that uses search tools to answer queries.

Implements tool_use pattern with Anthropic Claude API (via HTTP requests).
Each agent type has access to different search tools:
- Visual Agent: visual_search only
- Text Agent: text_search only
- Hybrid Agent: hybrid_search (RRF)
"""
import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    """A single step in the agent's reasoning process."""
    step_type: str  # "thinking", "tool_call", "tool_result", "answer"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_result: Optional[Any] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AgentResult:
    """Final result from an agent run."""
    answer: str
    thought_log: List[ThoughtStep] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    total_time_ms: float = 0.0
    tokens_used: int = 0
    agent_type: str = ""


# Tool definitions for Claude API
VISUAL_SEARCH_TOOL = {
    "name": "visual_search",
    "description": (
        "Search documents using visual multi-vector similarity (MaxSim). "
        "Best for finding visually similar content like charts, diagrams, layouts, "
        "and page compositions. Uses Jina V4 multi-vector embeddings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find visually relevant documents"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

TEXT_SEARCH_TOOL = {
    "name": "text_search",
    "description": (
        "Search documents using BM25 text matching. "
        "Best for finding specific keywords, phrases, and factual content. "
        "Uses Docling OCR-extracted text indexed with Elasticsearch BM25."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find text-matching documents"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

HYBRID_SEARCH_TOOL = {
    "name": "hybrid_search",
    "description": (
        "Search documents using Hybrid RRF (Reciprocal Rank Fusion) combining "
        "visual and text search. Best for comprehensive search that leverages "
        "both visual similarity and text matching. Uses Elasticsearch RRF retriever."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

# Agent type to tool mapping
AGENT_TOOLS = {
    "visual": [VISUAL_SEARCH_TOOL],
    "text": [TEXT_SEARCH_TOOL],
    "hybrid": [HYBRID_SEARCH_TOOL],
}

# System prompts per agent type
AGENT_SYSTEM_PROMPTS = {
    "visual": (
        "You are a Visual Search Agent. You find relevant documents by analyzing "
        "visual similarity using multi-vector embeddings (MaxSim). "
        "You can only use the visual_search tool. "
        "Analyze the user's query, decide on the best search strategy, "
        "then use your tool to find results. "
        "After getting results, provide a concise summary of what you found "
        "and why these results are relevant. "
        "Think step by step about your search strategy."
    ),
    "text": (
        "You are a Text Search Agent. You find relevant documents by matching "
        "keywords and phrases using BM25 text search on OCR-extracted content. "
        "You can only use the text_search tool. "
        "Analyze the user's query, consider keyword variations, "
        "then use your tool to find results. "
        "After getting results, provide a concise summary of what you found "
        "and why these results are relevant. "
        "Think step by step about your search strategy."
    ),
    "hybrid": (
        "You are a Hybrid Search Agent. You find relevant documents using "
        "Reciprocal Rank Fusion (RRF), which combines visual similarity "
        "and text matching for the best of both worlds. "
        "You can only use the hybrid_search tool. "
        "Analyze the user's query, then use your tool to find results. "
        "After getting results, provide a concise summary explaining "
        "how the hybrid approach found relevant results. "
        "Think step by step about your search strategy."
    ),
}


class SearchAgent:
    """
    LLM-powered search agent using Claude API with tool_use.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-20250514"
    MAX_TURNS = 5  # Maximum tool use turns

    def __init__(
        self,
        agent_type: str,
        api_key: str,
        search_fn: Callable,
    ):
        """
        Args:
            agent_type: "visual", "text", or "hybrid"
            api_key: Anthropic API key
            search_fn: Function to call for search (takes query, num_results)
        """
        if agent_type not in AGENT_TOOLS:
            raise ValueError(f"Unknown agent type: {agent_type}")

        self.agent_type = agent_type
        self.api_key = api_key
        self.search_fn = search_fn
        self.tools = AGENT_TOOLS[agent_type]
        self.system_prompt = AGENT_SYSTEM_PROMPTS[agent_type]

    def _call_api(self, messages: List[Dict], max_tokens: int = 1024) -> Dict:
        """Call Anthropic Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.MODEL,
            "max_tokens": max_tokens,
            "system": self.system_prompt,
            "tools": self.tools,
            "messages": messages,
        }

        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Dict:
        """Execute a search tool and return results."""
        query = tool_input.get("query", "")
        num_results = tool_input.get("num_results", 5)
        num_results = min(max(1, num_results), 10)

        results = self.search_fn(query, num_results)

        # Format results for LLM consumption
        formatted = []
        for i, r in enumerate(results, 1):
            entry = {
                "rank": i,
                "file_name": r.get("file_name", "Unknown"),
                "page_number": r.get("page_number", 0),
                "score": round(float(r.get("score", 0)), 4),
            }
            # Include text snippet if available
            text = r.get("text_content") or r.get("highlight") or ""
            if text:
                entry["text_snippet"] = text[:300]
            formatted.append(entry)

        return {"results": formatted, "count": len(formatted)}

    def run(self, user_query: str) -> Generator[ThoughtStep, None, AgentResult]:
        """
        Run the agent on a query, yielding thought steps as they happen.

        Usage:
            agent = SearchAgent("visual", api_key, search_fn)
            gen = agent.run("find financial reports")
            steps = []
            for step in gen:
                steps.append(step)
                # Update UI with step
            result = gen.value  # Final AgentResult (not available via simple iteration)
        """
        start_time = time.time()
        thought_log = []
        search_results = []
        total_tokens = 0

        messages = [{"role": "user", "content": user_query}]

        for turn in range(self.MAX_TURNS):
            try:
                response = self._call_api(messages)
            except Exception as e:
                error_step = ThoughtStep(
                    step_type="error",
                    content=f"API call failed: {str(e)}"
                )
                thought_log.append(error_step)
                yield error_step
                break

            total_tokens += response.get("usage", {}).get("input_tokens", 0)
            total_tokens += response.get("usage", {}).get("output_tokens", 0)
            stop_reason = response.get("stop_reason", "end_turn")

            # Process response content blocks
            content_blocks = response.get("content", [])
            assistant_content = []

            for block in content_blocks:
                if block["type"] == "text":
                    text = block["text"]
                    assistant_content.append(block)

                    # Is this intermediate thinking or final answer?
                    if stop_reason == "tool_use":
                        step = ThoughtStep(
                            step_type="thinking",
                            content=text
                        )
                    else:
                        step = ThoughtStep(
                            step_type="answer",
                            content=text
                        )
                    thought_log.append(step)
                    yield step

                elif block["type"] == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]
                    assistant_content.append(block)

                    # Log the tool call
                    call_step = ThoughtStep(
                        step_type="tool_call",
                        content=f"Calling {tool_name}",
                        tool_name=tool_name,
                        tool_input=tool_input
                    )
                    thought_log.append(call_step)
                    yield call_step

                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_input)
                    search_results.extend(tool_result.get("results", []))

                    # Log the result
                    result_step = ThoughtStep(
                        step_type="tool_result",
                        content=f"Got {tool_result['count']} results",
                        tool_name=tool_name,
                        tool_result=tool_result
                    )
                    thought_log.append(result_step)
                    yield result_step

                    # Add tool result to messages for next turn
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        }]
                    })
                    assistant_content = []  # Reset for next turn

            # If no tool_use, we're done
            if stop_reason != "tool_use":
                break

        elapsed = (time.time() - start_time) * 1000

        # Extract final answer
        final_answer = ""
        for step in reversed(thought_log):
            if step.step_type == "answer":
                final_answer = step.content
                break
        if not final_answer:
            final_answer = "No answer generated."

        result = AgentResult(
            answer=final_answer,
            thought_log=thought_log,
            search_results=search_results,
            total_time_ms=elapsed,
            tokens_used=total_tokens,
            agent_type=self.agent_type
        )

        return result

    def run_sync(self, user_query: str) -> AgentResult:
        """
        Synchronous version that collects all steps and returns final result.
        """
        start_time = time.time()
        thought_log = []
        search_results = []
        total_tokens = 0

        messages = [{"role": "user", "content": user_query}]

        for turn in range(self.MAX_TURNS):
            try:
                response = self._call_api(messages)
            except Exception as e:
                thought_log.append(ThoughtStep(
                    step_type="error",
                    content=f"API call failed: {str(e)}"
                ))
                break

            total_tokens += response.get("usage", {}).get("input_tokens", 0)
            total_tokens += response.get("usage", {}).get("output_tokens", 0)
            stop_reason = response.get("stop_reason", "end_turn")

            content_blocks = response.get("content", [])
            assistant_content = []

            for block in content_blocks:
                if block["type"] == "text":
                    assistant_content.append(block)
                    step_type = "thinking" if stop_reason == "tool_use" else "answer"
                    thought_log.append(ThoughtStep(
                        step_type=step_type,
                        content=block["text"]
                    ))

                elif block["type"] == "tool_use":
                    tool_name = block["name"]
                    tool_input = block["input"]
                    tool_id = block["id"]
                    assistant_content.append(block)

                    thought_log.append(ThoughtStep(
                        step_type="tool_call",
                        content=f"Calling {tool_name}",
                        tool_name=tool_name,
                        tool_input=tool_input
                    ))

                    tool_result = self._execute_tool(tool_name, tool_input)
                    search_results.extend(tool_result.get("results", []))

                    thought_log.append(ThoughtStep(
                        step_type="tool_result",
                        content=f"Got {tool_result['count']} results",
                        tool_name=tool_name,
                        tool_result=tool_result
                    ))

                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        }]
                    })
                    assistant_content = []

            if stop_reason != "tool_use":
                break

        elapsed = (time.time() - start_time) * 1000

        final_answer = ""
        for step in reversed(thought_log):
            if step.step_type == "answer":
                final_answer = step.content
                break
        if not final_answer:
            final_answer = "No answer generated."

        return AgentResult(
            answer=final_answer,
            thought_log=thought_log,
            search_results=search_results,
            total_time_ms=elapsed,
            tokens_used=total_tokens,
            agent_type=self.agent_type
        )


class AgentBattleArena:
    """
    Run multiple agents in parallel on the same query and compare results.
    """

    def __init__(
        self,
        api_key: str,
        visual_search_fn: Callable,
        text_search_fn: Callable,
        hybrid_search_fn: Callable,
    ):
        self.api_key = api_key
        self.agents = {
            "visual": SearchAgent("visual", api_key, visual_search_fn),
            "text": SearchAgent("text", api_key, text_search_fn),
            "hybrid": SearchAgent("hybrid", api_key, hybrid_search_fn),
        }

    def battle(
        self,
        query: str,
        agent_types: List[str] = None
    ) -> Dict[str, AgentResult]:
        """
        Run agents on the same query and return all results.

        Args:
            query: User's search query
            agent_types: Which agents to run (default: all three)

        Returns:
            Dict mapping agent_type to AgentResult
        """
        if agent_types is None:
            agent_types = ["visual", "text", "hybrid"]

        results = {}
        for agent_type in agent_types:
            if agent_type in self.agents:
                logger.info(f"Running {agent_type} agent on query: {query}")
                result = self.agents[agent_type].run_sync(query)
                results[agent_type] = result
                logger.info(
                    f"{agent_type} agent: {len(result.search_results)} results, "
                    f"{result.total_time_ms:.0f}ms, {result.tokens_used} tokens"
                )

        return results
