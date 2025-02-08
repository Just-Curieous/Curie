# Notes for gpt researcher setup:
# Some env configs are stored in ../setup/env.sh and some stored in assets/gpt_research_config.json
"""
Use the following in miniconda3/envs/langgraph/lib/python3.11/site-packages/gpt_researcher/memory/embeddings.py

_embeddings = AzureOpenAIEmbeddings(
                    model=model,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    openai_organization=os.getenv('OPENAI_ORGANIZATION'),
                    **embdding_kwargs,
)

Use the following in miniconda3/envs/langgraph/lib/python3.11/site-packages/gpt_researcher/llm_provider/generic/base.py

around line 52:

llm = AzureChatOpenAI(
                # model='gpt-4o',
                api_key=os.getenv('OPENAI_API_KEY'),
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version=os.getenv('API_VERSION'),
                organization=os.getenv('OPENAI_ORGANIZATION'),
                **kwargs
)

"""

from gpt_researcher import GPTResearcher
import asyncio

async def get_report(query: str, report_type: str):
    researcher = GPTResearcher(query, report_type, config_path="assets/gpt_research_config.json")
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    
    # Get additional information
    research_context = researcher.get_research_context()
    research_costs = researcher.get_costs()
    research_images = researcher.get_research_images()
    research_sources = researcher.get_research_sources()
    
    return report, research_context, research_costs, research_images, research_sources

if __name__ == "__main__":
    query = "what team may win the NBA finals?"
    report_type = "research_report"

    report, context, costs, images, sources = asyncio.run(get_report(query, report_type))
    
    print("Report:")
    print(report)
    print("\nResearch Costs:")
    print(costs)
    print("\nNumber of Research Images:")
    print(len(images))
    print("\nNumber of Research Sources:")
    print(len(sources))