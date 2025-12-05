from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow import Step, Workflow
from dotenv import load_dotenv

# load all the api keys to env
load_dotenv()

# define a model
model = OpenAIChat(id="gpt-4o-mini")

# ========================== models ====================================
# essay writing agent
essay_writing_agent = Agent(
    id="essay-writing-agent",
    name="Essay Writing Agent",
    model=model,
    instructions=["You are expert in writing an essay",
                  "Write a well structured essay on a given topic",
                  "Limit your response to max 400 words"]
)

# Extraction agent to extract imp points from the essay
extraction_agent = Agent(
    id="extraction-agent",
    name="Extraction Agent",
    model=model,
    instructions=["You are an expert in extracting important points from the essay",
                  "Summarize a key points in a consise manner",
                  "Your output should be in good format"]
)


# =============================== step =================================
essay_writing_step = Step(
    name="Essay writing step",
    agent=essay_writing_agent,
    description="Generates Essay from the topic given by the user"
)

extraction_step = Step(
    name="Extraction Step",
    agent=extraction_agent,
    description="Extracts important points from the essay generated in the previous step"
)

# =================================== workflow =================================

workflow = Workflow(
    id="essay-workflow",
    name="Essay writing and point extracting worklow",
    steps=[essay_writing_step, extraction_step],
    description="A workflow that first writes an essay on a given topic and then extracts important points from that essay"
)


# Execite and run the workflow
workflow.print_response(input="topic : Impact of AI on today's industry",
                        stream=True, markdown=True)
