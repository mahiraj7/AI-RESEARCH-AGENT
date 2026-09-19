# ============================================================
# AI RESEARCH AGENT
# ============================================================

from typing import TypedDict

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)

from langchain_ollama import ChatOllama

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.prebuilt import ToolNode

from tools import (
    get_company_info,
    get_stock_info,
    research_company,
    web_search,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "qwen2.5:7b"

OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# 2. LLM
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    base_url=OLLAMA_BASE_URL,
)


# ============================================================
# 3. TOOLS
# ============================================================

tools = [
    get_company_info,
    get_stock_info,
    research_company,
    web_search,
]


# Bind tools to LLM
llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 4. AGENT STATE
# ============================================================

class AgentState(TypedDict):

    messages: list


# ============================================================
# 5. LLM NODE
# ============================================================

def llm_node(state: AgentState):
    """Run the LLM and allow it to decide whether a tool is needed."""

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": (
            state["messages"]
            + [response]
        )
    }


# ============================================================
# 6. TOOL ROUTING
# ============================================================

def should_continue(state: AgentState):
    """Route to tools if the LLM requested a tool call."""

    last_message = state["messages"][-1]

    if getattr(
        last_message,
        "tool_calls",
        None
    ):
        return "tool"

    return "end"


# ============================================================
# 7. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 8. AFTER TOOL ROUTING
# ============================================================

def after_tool(state: AgentState):
    """Decide whether tool results need research processing or another LLM step."""

    last_message = state["messages"][-1]

    if isinstance(
        last_message,
        ToolMessage
    ):

        if last_message.name == "web_search":
            return "research"

        if last_message.name == "research_company":
            return "research"

    return "llm"


# ============================================================
# 9. RESEARCH NODE
# ============================================================

def research_node(state: AgentState):
    """Analyze web-search results and create a structured research report."""

    messages = state["messages"]

    tool_results = []

    for message in messages:

        if isinstance(
            message,
            ToolMessage
        ):

            tool_results.append(
                str(message.content)
            )

    research_context = "\n\n".join(
        tool_results
    )


    # --------------------------------------------------------
    # Research prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI research analyst.

Analyze the research data below and create
a clear, concise and professional research report.

Use ONLY the information provided in the
research data.

Structure the report exactly like this:

# Research Report

## Overview

Give a short factual overview.

## Key Findings

List the most important findings.

## Latest Developments

Summarize the important developments
found in the research data.

## Sources

List the sources in this format:

- Source Title: URL

Rules:

- Do not invent facts.
- Do not use outside knowledge.
- Use factual and neutral language.
- Do not exaggerate claims.
- Do not use promotional language.
- Do not use words such as "best",
  "unparalleled", "game-changing",
  or similar promotional terms.
- Preserve source URLs.
- Do not create Markdown links.
- Do not duplicate URLs.
- Keep the report concise.
- Only include information supported
  by the research data.

Research Data:

{research_context}
"""


    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    response = llm.invoke(
        prompt
    )


    return {
        "messages": (
            messages
            + [response]
        )
    }


# ============================================================
# 10. BUILD GRAPH
# ============================================================

graph_builder = StateGraph(
    AgentState
)


# ============================================================
# 11. ADD NODES
# ============================================================

graph_builder.add_node(
    "llm",
    llm_node
)

graph_builder.add_node(
    "tool",
    tool_node
)

graph_builder.add_node(
    "research",
    research_node
)


# ============================================================
# 12. GRAPH EDGES
# ============================================================

# START → LLM

graph_builder.add_edge(
    START,
    "llm"
)


# ------------------------------------------------------------
# LLM → TOOL OR END
# ------------------------------------------------------------

graph_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tool": "tool",
        "end": END,
    }
)


# ------------------------------------------------------------
# TOOL → RESEARCH OR LLM
# ------------------------------------------------------------

graph_builder.add_conditional_edges(
    "tool",
    after_tool,
    {
        "research": "research",
        "llm": "llm",
    }
)


# ------------------------------------------------------------
# RESEARCH → END
# ------------------------------------------------------------

graph_builder.add_edge(
    "research",
    END
)


# ============================================================
# 13. COMPILE GRAPH
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 14. RUN AGENT
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("         AI RESEARCH AGENT")
    print("========================================")
    print()

    query = input(
        "Ask your research question: "
    )


    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ]
        }
    )


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    print()
    print("========================================")
    print("             FINAL ANSWER")
    print("========================================")
    print()

    final_message = result[
        "messages"
    ][-1]

    print(
        final_message.content
    )

    print()
    print("========================================")
