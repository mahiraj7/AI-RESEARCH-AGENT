# ============================================================
# AI RESEARCH AGENT - TOOLS
# ============================================================

from ddgs import DDGS
from langchain_core.tools import tool


# ============================================================
# 1. COMPANY INFORMATION TOOL
# ============================================================

@tool
def get_company_info(company: str):
    """Get basic information about a company."""

    company_data = {
        "NVIDIA": {
            "name": "NVIDIA",
            "industry": "Semiconductors and AI",
            "products": (
                "GPUs, AI platforms, "
                "data-center technologies"
            ),
        },
        "Microsoft": {
            "name": "Microsoft",
            "industry": "Technology",
            "products": (
                "Cloud computing, software, "
                "AI and productivity tools"
            ),
        },
        "Apple": {
            "name": "Apple",
            "industry": "Technology",
            "products": (
                "iPhone, Mac, iPad, "
                "software and services"
            ),
        },
        "Google": {
            "name": "Google",
            "industry": "Technology",
            "products": (
                "Search, cloud computing, "
                "AI and advertising"
            ),
        },
        "Amazon": {
            "name": "Amazon",
            "industry": "Technology and E-commerce",
            "products": (
                "E-commerce, AWS cloud services, "
                "advertising and AI"
            ),
        },
    }

    key = company.strip().upper()

    if key in company_data:
        return company_data[key]

    return {
        "name": company,
        "message": (
            "Basic information is not available "
            "in the local company database."
        )
    }


# ============================================================
# 2. STOCK INFORMATION TOOL
# ============================================================

@tool
def get_stock_info(company: str):
    """Get stock ticker and exchange information for a company."""

    stock_data = {
        "NVIDIA": {
            "name": "NVIDIA",
            "ticker": "NVDA",
            "exchange": "NASDAQ",
        },
        "MICROSOFT": {
            "name": "Microsoft",
            "ticker": "MSFT",
            "exchange": "NASDAQ",
        },
        "APPLE": {
            "name": "Apple",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
        },
        "AMAZON": {
            "name": "Amazon",
            "ticker": "AMZN",
            "exchange": "NASDAQ",
        },
        "GOOGLE": {
            "name": "Alphabet",
            "ticker": "GOOGL",
            "exchange": "NASDAQ",
        },
    }

    key = company.strip().upper()

    if key in stock_data:
        return stock_data[key]

    return {
        "name": company,
        "message": (
            "Stock information is not available "
            "in the local database."
        )
    }


# ============================================================
# 3. COMPANY RESEARCH TOOL
# ============================================================

@tool
def research_company(company: str):
    """Search the web for general research information about a company."""

    query = f"{company} company latest information"

    return _perform_search(
        query=query,
        max_results=5
    )


# ============================================================
# 4. WEB SEARCH TOOL
# ============================================================

@tool
def web_search(query: str):
    """Search the web for information related to a query."""

    return _perform_search(
        query=query,
        max_results=5
    )


# ============================================================
# 5. INTERNAL SEARCH FUNCTION
# ============================================================

def _perform_search(
    query: str,
    max_results: int = 5
):
    """Perform a DuckDuckGo web search and return structured results."""

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=max_results
            )

            for result in search_results:

                title = result.get(
                    "title",
                    ""
                )

                url = result.get(
                    "href",
                    ""
                )

                snippet = result.get(
                    "body",
                    ""
                )

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                })

    except Exception as e:

        return {
            "error": f"Web search failed: {str(e)}",
            "results": []
        }

    return {
        "query": query,
        "results": results
    }


# ============================================================
# 6. TEST TOOLS DIRECTLY
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("Testing get_company_info")
    print("==============================")

    print(
        get_company_info.invoke(
            {"company": "NVIDIA"}
        )
    )

    print("\n==============================")
    print("Testing get_stock_info")
    print("==============================")

    print(
        get_stock_info.invoke(
            {"company": "NVIDIA"}
        )
    )

    print("\n==============================")
    print("Testing web_search")
    print("==============================")

    print(
        web_search.invoke(
            {
                "query": (
                    "NVIDIA latest AI developments"
                )
            }
        )
    )
