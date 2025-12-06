from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.arxiv import ArxivTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Step, Workflow, Parallel
from dotenv import load_dotenv

# load all the keys in the env
load_dotenv()

# define tool
arxiv_tool = ArxivTools()

duckduckgo_tool = DuckDuckGoTools()

hackernews_tool = HackerNewsTools()

# define model
model = OpenAIChat(id="gpt-4o-mini")
# ========================== Agents ===================================
arxiv_agent = Agent(
    id="arxiv-agent",
    name="Arxiv Agent",
    model=model,
    tools=[arxiv_tool],
    instructions=["You are an expert research agent",
                  "provide accurate and relavent research paper based on the user input",
                  "add heading to the output source : Arxiv Search"]
)

duckduckgo_agent = Agent(
    id="duckduckgo-agent",
    name="duckduckgo agent",
    model=model,
    tools={duckduckgo_tool},
    instructions=["You are an expert web search agent using duckduckgo search",
                  "provide accurate and relevent result based on the user input",
                  "add heading to the output source : Duckduckgo Search"]
)


hackernews_agent = Agent(
    id="hackernews-agent",
    name="Hackernews agent",
    model=model,
    tools={hackernews_tool},
    instructions=[
        "You are an expert in retrieving trending topics and latest news from Hacker news",
        "Add heading to the output source : Hackernews Search"]
)

# report generation agent
report_generation_agent = Agent(
    id="report-generation-agent",
    name="Report Generation Agent",
    model=model,
    instructions=["You are an expert in report generation",
                  "Compile all the information from the all three (arxiv, duckduckgo, hackernews) sources into coherent and comprehesive report",
                  "use the information from the all source (arxiv agent, duckduckgo agent, hackernews agent)",
                  "mention the source of the information",
                  "the output should be in proper format"],
    markdown=True
)


# ================================= Steps ==========================================
# arxiv search step
arxiv_search_step = Step(
    name="Arxiv Search Step",
    agent=arxiv_agent,
    description="Performs research using arxiv search"
)

# duckduckgo search step
duckduckgo_search_step = Step(
    name="Duckduckgo Search Step",
    agent=duckduckgo_agent,
    description="Performs web search using duckduckgo search"
)

# hackernews search step
hackernews_search_step = Step(
    name="Hackernews Search Step",
    agent=hackernews_agent,
    description="Performs the latest news search using hackernews search"
)

# report generation step
report_generation_step = Step(
    name="Report Generation Step",
    agent=report_generation_agent,
    description="Generates well formatted report using the all gathered information from all three(arxiv, duckduckgo, hackernews) source"
)

# parallel step
parallel_step = Parallel(
    arxiv_search_step, duckduckgo_search_step, hackernews_search_step,
    name="Parallel Search Step",
    description="Perform searches from the various source parellely"
)

# =============================== Wrokflow ========================================
parallel_workflow = Workflow(
    id="parallel workflow",
    name="Parallel Wrokflow",
    steps=[parallel_step, report_generation_step],
    description="A workflow that performs web searching using multiple agents in parallel and then generates a report based on the retrieved information"
)

parallel_workflow.print_response(input="topic : transformer",
                                 stream=True,
                                 markdown=True)
