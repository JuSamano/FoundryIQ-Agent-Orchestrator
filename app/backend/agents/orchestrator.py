"""
Multi-Agent Orchestrator with KB Grounding.

Routes queries to specialized agents:
- HR Agent → kb1-hr (policies, PTO, benefits)
- Marketing Agent → kb2-marketing (campaigns, brand, analytics)
- Products Agent → kb3-products (catalog, specs, pricing)
"""

import asyncio
import os
from azure.identity.aio import DefaultAzureCredential
from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureAIAgentClient, AzureAISearchContextProvider

# Configuration
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://srch-g5mlw6gto4s6i.search.windows.net")
PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT", "https://jusamano-2099-resource.services.ai.azure.com/api/projects/jusamano-2099")
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# Agent instructions
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


async def route_query(client: ChatAgent, query: str) -> str:
    """Route a query to the appropriate specialist."""
    message = ChatMessage(role=Role.USER, text=query)
    response = await client.run(message)
    route = response.text.strip().lower()
    
    # Normalize routing
    if "hr" in route:
        return "hr"
    elif "marketing" in route or "brand" in route or "campaign" in route:
        return "marketing"
    elif "product" in route:
        return "products"
    else:
        return "hr"  # Default


async def run_orchestrator():
    """Run the multi-agent orchestrator."""
    
    credential = DefaultAzureCredential()
    
    async with (
        AzureAIAgentClient(
            project_endpoint=PROJECT_ENDPOINT,
            model_deployment_name=MODEL,
            credential=credential,
        ) as client,
        AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            knowledge_base_name="kb1-hr",
            credential=credential,
            mode="agentic",
            knowledge_base_output_mode="answer_synthesis",
        ) as hr_kb,
        AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            knowledge_base_name="kb2-marketing",
            credential=credential,
            mode="agentic",
            knowledge_base_output_mode="answer_synthesis",
        ) as marketing_kb,
        AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            knowledge_base_name="kb3-products",
            credential=credential,
            mode="agentic",
            knowledge_base_output_mode="answer_synthesis",
        ) as products_kb,
    ):
        # Create router agent (no KB, just for routing decisions)
        router = ChatAgent(
            chat_client=client,
            instructions=ROUTER_INSTRUCTIONS,
        )
        
        # Create specialist agents with KB grounding
        hr_agent = ChatAgent(
            chat_client=client,
            context_provider=hr_kb,
            instructions=HR_INSTRUCTIONS,
        )
        
        marketing_agent = ChatAgent(
            chat_client=client,
            context_provider=marketing_kb,
            instructions=MARKETING_INSTRUCTIONS,
        )
        
        products_agent = ChatAgent(
            chat_client=client,
            context_provider=products_kb,
            instructions=PRODUCTS_INSTRUCTIONS,
        )
        
        specialists = {
            "hr": hr_agent,
            "marketing": marketing_agent,
            "products": products_agent,
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
                if query.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break
                
                # Route the query
                route = await route_query(router, query)
                print(f"📍 Routing to: {route.upper()} agent")
                
                # Get specialist response
                agent = specialists[route]
                message = ChatMessage(role=Role.USER, text=query)
                response = await agent.run(message)
                
                print(f"\n💬 Response:\n{response.text}\n")
                print("-" * 55)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    await credential.close()


async def run_single_query(query: str) -> tuple[str, str, list[dict]]:
    """Run a single query and return (route, response, sources)."""
    
    credential = DefaultAzureCredential()
    
    kb_map = {
        "hr": "kb1-hr",
        "marketing": "kb2-marketing",
        "products": "kb3-products",
    }
    
    async with (
        AzureAIAgentClient(
            project_endpoint=PROJECT_ENDPOINT,
            model_deployment_name=MODEL,
            credential=credential,
        ) as client,
        AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            knowledge_base_name="kb1-hr",
            credential=credential,
            mode="agentic",
            knowledge_base_output_mode="answer_synthesis",
        ) as hr_kb,
        AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            knowledge_base_name="kb2-marketing",
            credential=credential,
            mode="agentic",
            knowledge_base_output_mode="answer_synthesis",
        ) as marketing_kb,
        AzureAISearchContextProvider(
            endpoint=SEARCH_ENDPOINT,
            knowledge_base_name="kb3-products",
            credential=credential,
            mode="agentic",
            knowledge_base_output_mode="answer_synthesis",
        ) as products_kb,
    ):
        router = ChatAgent(chat_client=client, instructions=ROUTER_INSTRUCTIONS)
        
        specialists = {
            "hr": ChatAgent(chat_client=client, context_provider=hr_kb, instructions=HR_INSTRUCTIONS),
            "marketing": ChatAgent(chat_client=client, context_provider=marketing_kb, instructions=MARKETING_INSTRUCTIONS),
            "products": ChatAgent(chat_client=client, context_provider=products_kb, instructions=PRODUCTS_INSTRUCTIONS),
        }
        
        route = await route_query(router, query)
        agent = specialists[route]
        message = ChatMessage(role=Role.USER, text=query)
        response = await agent.run(message)
        
        # Extract sources from citations if available
        sources = []
        kb_name = kb_map.get(route, "unknown")
        
        # Try to get citations from the response
        if hasattr(response, 'citations') and response.citations:
            for citation in response.citations:
                source_info = {"kb": kb_name}
                if hasattr(citation, 'title') and citation.title:
                    source_info["title"] = citation.title
                if hasattr(citation, 'filepath') and citation.filepath:
                    source_info["filepath"] = citation.filepath
                if hasattr(citation, 'url') and citation.url:
                    source_info["url"] = citation.url
                if hasattr(citation, 'chunk_id') and citation.chunk_id:
                    source_info["chunk_id"] = citation.chunk_id
                if len(source_info) > 1:  # Has more than just kb
                    sources.append(source_info)
        
        # Try context attribute
        if not sources and hasattr(response, 'context') and response.context:
            for ctx in response.context:
                source_info = {"kb": kb_name}
                if hasattr(ctx, 'title'):
                    source_info["title"] = ctx.title
                if hasattr(ctx, 'source'):
                    source_info["filepath"] = ctx.source
                if len(source_info) > 1:
                    sources.append(source_info)
        
        # Try grounding_data if available
        if not sources and hasattr(response, 'grounding_data') and response.grounding_data:
            for data in response.grounding_data:
                source_info = {"kb": kb_name}
                if hasattr(data, 'title'):
                    source_info["title"] = data.title
                if hasattr(data, 'filepath'):
                    source_info["filepath"] = data.filepath
                if len(source_info) > 1:
                    sources.append(source_info)
        
        # Default sources if nothing found
        if not sources:
            # Provide default document names based on KB
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
        
        return route, response.text, sources
    
    await credential.close()


if __name__ == "__main__":
    asyncio.run(run_orchestrator())
