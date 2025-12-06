from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow import Step, Workflow, Loop, StepOutput
from dotenv import load_dotenv

# load all keys
load_dotenv()

# create a model
model = OpenAIChat(id="gpt-4o-mini")


def word_count_condition(step_output: StepOutput) -> bool:
    """Condition to check if the story is less than 300 words"""
    # check if there is output from the prev step
    if step_output:
        for output in step_output:
            story_content: str = output.content
            word_count: int = len(story_content.split(" "))
            if word_count <= 300:
                return True
            else:
                return False
    else:
        return False


# ============================== agents ====================================
# story generation agent
story_generation_agent = Agent(
    id="story-generation-agent",
    name="Story generation agent",
    instructions=["You are an expert story writter",
                  "Create engaging and imaginative short stories based on user request",
                  "stick to the word limit requested by user"],
    model=model
)


# =========================== steps ===========================================
# create agent step
story_generation_step = Step(
    name="Story Generation Step",
    agent=story_generation_agent,
    description="Generates a short story based on user's prompt"
)

# define the looping step
looping_step = Loop(
    steps=[story_generation_step],
    name="Story generation step",
    description="Generates stories in loop till condition is met",
    end_condition=word_count_condition
)

# ========================= workflow ====================================
workflow = Workflow(
    id="story-generation-wrokflow",
    name="Story Generation workflow",
    steps=[looping_step],
    description="A workflow generates a short stories and ensure they are less than 300 words using looping mechanism"
)


# input to workflow
workflow.print_response(input="Title : A magical advanture in a castle",
                        stream=True,
                        markdown=True)
