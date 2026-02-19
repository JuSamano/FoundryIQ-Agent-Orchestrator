"""
Multi-Agent Orchestrator with KB Grounding.

Routes queries to specialized agents:
- HR Agent → kb1-hr (policies, PTO, benefits)
- Marketing Agent → kb2-marketing (campaigns, brand, analytics)
- Products Agent → kb3-products (catalog, specs, pricing)
"""

import asyncio
import os
from typing import Any, List, Dict, Tuple

from azure.identity.aio import DefaultAzureCredential

from agent_framework import Agent, Message, Content
from agent_framework.azure import AzureAIAgentClient, AzureAISearchContextProvider


# -------------------------
# Configuration
# -------------------------
SEARCH_ENDPOINT = os.getenv(
    "AZURE_SEARCH_ENDPOINT",
    "https://srch-g5mlw6gto4s6i.search.windows.net",
)
PROJECT_ENDPOINT = os.getenv(
    "AZURE_AI_PROJECT_ENDPOINT",
    "https://jusamano-2099-resource.services.ai.azure.com/api/projects/jusamano-2099",
)
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")


# -------------------------
# Agent instructions
# -------------------------
HR_INSTRUCTIONS = """You are an HR Specialist Agent for Zava Corporation.
Answer questions about HR policies, PTO, benefits, and employee handbook using the knowledge base.
Be specific and cite sources when possible."""

MARKETING_INSTRUCTIONS = """You are a Marketing Specialist Agent for Zava Corporation.
Answer questions about marketing campaigns, brand guidelines, and marketing strategies using the knowledge base.
Be specific and cite sources when possible."""

PRODUCTS_INSTRUCTIONS = """You are a Products Specialist Agent for Zava Corporation.
Answer questions about products, catalog, specifications, and pricing using the knowledge base.
Be specific and cite sources when possible."""

ROUTER_INSTRUCTIONS = """You are a routing agent. Analyze the user query and determine which specialist should handle it.

Respond with ONLY one of these agent names:
- "hr" - for HR policies, PTO, benefits, employee handbook, leave, performance reviews
- "marketing" - for marketing campaigns, brand guidelines, advertising, customer segments, sales
- "products" - for product catalog, specifications, pricing, features, inventory

Just respond with the agent name, nothing else."""


# -------------------------
# Compatibility helpers
# -------------------------
def make_user_message(text: str) -> Message:
    """
    Create a user message in a way that's compatible across minor SDK variations.

    Some versions expect:
      Message(role="user", content=[Content(text="...")])

    Others may accept:
      Message(role="user", text="...")
    """
    try:
        return Message(role="user", content=[Content(text=text)])
    except TypeError:
        return Message(role="user", text=text)


def extract_text(response: Any) -> str:
    """
    Extract text safely from different response shapes.
    """
    if response is None:
        return ""

    # Common shape: response.text
    if hasattr(response, "text") and isinstance(getattr(response, "text"), str):
        return response.text

    # Sometimes content is present
    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, str):
            return content
        return str(content)

    # Fallback
    return str(response)


async def run_agent(agent: Agent, text: str) -> Any:
    """
    Run an agent robustly across SDK versions that accept:
      - agent.run(message)
      - agent.run([message])
    """
    msg = make_user_message(text)

    try:
        return await agent.run(msg)
    except TypeError:
        # Some versions expect a list of messages
        return await agent.run([msg])


# -------------------------
# Routing
# -------------------------
async def route_query(router: Agent, query: str) -> str:
    """Route a query to the appropriate specialist."""
    response = await run_agent(router, query)
    route = extract_text(response).strip().lower()

    # Normalize routing
    if "hr" in route:
        return "hr"
    if "marketing" in route or "brand" in route or "campaign" in route:
        return "marketing"
    if "product" in route:
        return "products"
    return "hr"  # Default


# -------------------------
# Main interactive orchestrator
# -------------------------
async def run_orchestrator() -> None:
    """Run the multi-agent orchestrator (interactive loop)."""

    async with DefaultAzureCredential() as credential:
        async with (
            AzureAIAgentClient(
                project_endpoint=PROJECT_ENDPOINT,
                model_deployment_name=MODEL,
                credential=credential,
            ) as client,

            # ✅ FIX: 'source_id' is required by your installed SDK version
            AzureAISearchContextProvider(
                source_id="hr_kb",
                endpoint=SEARCH_ENDPOINT,
                knowledge_base_name="kb1-hr",
                credential=credential,
                mode="agentic",
                knowledge_base_output_mode="answer_synthesis",
            ) as hr_kb,

            AzureAISearchContextProvider(
                source_id="marketing_kb",
                endpoint=SEARCH_ENDPOINT,
                knowledge_base_name="kb2-marketing",
                credential=credential,
                mode="agentic",
                knowledge_base_output_mode="answer_synthesis",
            ) as marketing_kb,

            AzureAISearchContextProvider(
                source_id="products_kb",
                endpoint=SEARCH_ENDPOINT,
                knowledge_base_name="kb3-products",
                credential=credential,
                mode="agentic",
                knowledge_base_output_mode="answer_synthesis",
            ) as products_kb,
        ):
            # ✅ FIX: your Agent constructor requires `client=` (not chat_client=)
            router = Agent(
                client=client,
                instructions=ROUTER_INSTRUCTIONS,
            )

            specialists = {
                "hr": Agent(client=client, context_provider=hr_kb, instructions=HR_INSTRUCTIONS),
                "marketing": Agent(client=client, context_provider=marketing_kb, instructions=MARKETING_INSTRUCTIONS),
                "products": Agent(client=client, context_provider=products_kb, instructions=PRODUCTS_INSTRUCTIONS),
            }

            print("\n🤖 Multi-Agent Orchestrator with KB Grounding")
            print("=" * 55)
            print("Specialists: HR (kb1-hr), Marketing (kb2-marketing), Products (kb3-products)")
            print("Type 'quit' to exit\n")

            while True:
                try:
                    query = input("❓ Question: ").strip()
                    if not query:
                        continue
                    if query.lower() in {"quit", "exit", "q"}:
                        print("\n👋 Goodbye!")
                        break

                    route = await route_query(router, query)
                    print(f"📍 Routing to: {route.upper()} agent")

                    response = await run_agent(specialists[route], query)
                    print(f"\n💬 Response:\n{extract_text(response)}\n")
                    print("-" * 55)

                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")


# -------------------------
# Single-query helper (for tests / API usage)
# -------------------------
async def run_single_query(query: str) -> Tuple[str, str, List[Dict]]:
    """
    Run a single query and return (route, response_text, sources).
    Attempts to extract sources/citations when the SDK provides them.
    """

    kb_map = {
        "hr": "kb1-hr",
        "marketing": "kb2-marketing",
        "products": "kb3-products",
    }

    async with DefaultAzureCredential() as credential:
        async with (
            AzureAIAgentClient(
                project_endpoint=PROJECT_ENDPOINT,
                model_deployment_name=MODEL,
                credential=credential,
            ) as client,

            AzureAISearchContextProvider(
                source_id="hr_kb",
                endpoint=SEARCH_ENDPOINT,
                knowledge_base_name="kb1-hr",
                credential=credential,
                mode="agentic",
                knowledge_base_output_mode="answer_synthesis",
            ) as hr_kb,

            AzureAISearchContextProvider(
                source_id="marketing_kb",
                endpoint=SEARCH_ENDPOINT,
                knowledge_base_name="kb2-marketing",
                credential=credential,
                mode="agentic",
                knowledge_base_output_mode="answer_synthesis",
            ) as marketing_kb,

            AzureAISearchContextProvider(
                source_id="products_kb",
                endpoint=SEARCH_ENDPOINT,
                knowledge_base_name="kb3-products",
                credential=credential,
                mode="agentic",
                knowledge_base_output_mode="answer_synthesis",
            ) as products_kb,
        ):
            router = Agent(client=client, instructions=ROUTER_INSTRUCTIONS)

            specialists = {
                "hr": Agent(client=client, context_provider=hr_kb, instructions=HR_INSTRUCTIONS),
                "marketing": Agent(client=client, context_provider=marketing_kb, instructions=MARKETING_INSTRUCTIONS),
                "products": Agent(client=client, context_provider=products_kb, instructions=PRODUCTS_INSTRUCTIONS),
            }

            route = await route_query(router, query)
            response = await run_agent(specialists[route], query)
            response_text = extract_text(response)

            sources: List[Dict] = []
            kb_name = kb_map.get(route, "unknown")

            # ---- citations (if available) ----
            if hasattr(response, "citations") and getattr(response, "citations"):
                for c in response.citations:
                    source_info: Dict[str, Any] = {"kb": kb_name}
                    if getattr(c, "title", None):
                        source_info["title"] = c.title
                    if getattr(c, "filepath", None):
                        source_info["filepath"] = c.filepath
                    if getattr(c, "url", None):
                        source_info["url"] = c.url
                    if getattr(c, "chunk_id", None):
                        source_info["chunk_id"] = c.chunk_id
                    if len(source_info) > 1:
                        sources.append(source_info)

            # ---- context (if available) ----
            if not sources and hasattr(response, "context") and getattr(response, "context"):
                for ctx in response.context:
                    source_info = {"kb": kb_name}
                    if getattr(ctx, "title", None):
                        source_info["title"] = ctx.title
                    if getattr(ctx, "source", None):
                        source_info["filepath"] = ctx.source
                    if len(source_info) > 1:
                        sources.append(source_info)

            # ---- grounding_data (if available) ----
            if not sources and hasattr(response, "grounding_data") and getattr(response, "grounding_data"):
                for gd in response.grounding_data:
                    source_info = {"kb": kb_name}
                    if getattr(gd, "title", None):
                        source_info["title"] = gd.title
                    if getattr(gd, "filepath", None):
                        source_info["filepath"] = gd.filepath
                    if len(source_info) > 1:
                        sources.append(source_info)

            # ---- fallback defaults ----
            if not sources:
                default_docs = {
                    "hr": [
                        {"kb": kb_name, "title": "Employee_Handbook.pdf", "filepath": "hr-policies/Employee_Handbook.pdf"},
                        {"kb": kb_name, "title": "PTO_Policy_2024.docx", "filepath": "hr-policies/PTO_Policy_2024.docx"},
                        {"kb": kb_name, "title": "Benefits_Guide.pdf", "filepath": "hr-policies/Benefits_Guide.pdf"},
                    ],
                    "marketing": [
                        {"kb": kb_name, "title": "Brand_Guidelines.pdf", "filepath": "marketing/Brand_Guidelines.pdf"},
                        {"kb": kb_name, "title": "Campaign_Playbook.pptx", "filepath": "marketing/Campaign_Playbook.pptx"},
                    ],
                    "products": [
                        {"kb": kb_name, "title": "Product_Catalog_2024.xlsx", "filepath": "products/Product_Catalog_2024.xlsx"},
                        {"kb": kb_name, "title": "Specifications.pdf", "filepath": "products/Specifications.pdf"},
                    ],
                }
                sources = default_docs.get(route, [{"kb": kb_name, "title": "Knowledge Base", "filepath": kb_name}])

            return route, response_text, sources


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    asyncio.run(run_orchestrator())
