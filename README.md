# AI Research Agent

An agentic AI research assistant built with LangGraph, LangChain, Ollama and Qwen 2.5.

The agent can understand a user's research request, decide which tools are required, perform web searches, collect information and generate a structured research report.

---

## Features

- LLM-powered agent using Qwen 2.5 7B
- LangGraph-based agent workflow
- Tool calling
- Web search using DuckDuckGo
- Company information tool
- Stock information tool
- Research synthesis
- Structured research reports
- Source URLs included in final responses
- Runs locally using Ollama

---

## Architecture

```text
User
  |
  v
LLM (Qwen 2.5)
  |
  v
Tool Decision
  |
  +--------------------+
  |                    |
  v                    v
Tool Required        No Tool
  |                    |
  v                    v
ToolNode              END
  |
  v
Web Search / Company / Stock
  |
  v
Research Node
  |
  v
LLM
  |
  v
Final Research Report