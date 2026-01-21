# Kibana Agent Builder Setup Guide

This guide explains how to connect the **Elastic Visual Comparison MCP Server** to Kibana Agent Builder.

## 1. Prerequisites (Local Development)
Since the MCP server runs locally (port 8000), you need to expose it to the internet so Elastic Cloud can access it.

1. Install **ngrok** (or similar tunneling tool).
2. Run the MCP server:
   ```bash
   poetry run python backend/mcp_server/run.py
   ```
3. In a separate terminal, start the tunnel:
   ```bash
   ngrok http 8000
   ```
4. Copy the HTTPS URL (e.g., `https://1234-abcd.ngrok-free.app`).

## 2. Create Agent in Kibana

1. Log in to your **Elastic Cloud** Kibana instance.
2. Navigate to: **Menu** -> **Search** -> **Agent Builder**.
3. Click **Create Agent**.
4. Name your agent (e.g., "Visual Search Analyst").

## 3. Register MCP Server

1. In the Agent configuration, look for the **MCP Servers** section.
2. Click **Add MCP Server**.
3. Enter the details:
   *   **Name:** `elastic-visual-comparison`
   *   **URL:** `<YOUR_NGROK_URL>/sse`
       *   Example: `https://1234-abcd.ngrok-free.app/sse`
4. Click **Connect** or **Verify**. You should see a success message indicating the tools (`compare_search_results`) are available.

## 4. Configure System Prompt

Copy and paste the following into the **System Prompt** / **Instructions** field:

```text
You are a Visual Search Analyst.
When a user asks a question, ALWAYS use the `compare_search_results` tool.
Do not answer from your own knowledge.
Present the findings in the following format:
1. **Visual Search Found:** (Summarize findings from visual index)
2. **Text Search Found:** (Summarize findings from text index)
3. **Analysis:** Compare which method was more accurate and explain why (e.g., "Visual search correctly identified the trend line in the chart, while text search only found the caption.").
```

## 5. Test

1. Save the Agent.
2. In the chat window, ask: "Compare results for 'red shoes in 2024 report'".
3. The Agent should call the tool and display the comparison table.
