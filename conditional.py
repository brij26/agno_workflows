from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
from agno.workflow import Step, Workflow, Condition, StepInput, StepOutput

load_dotenv()

model = OpenAIChat(id="gpt-4o-mini")


def review_email_condition(step_input: StepInput) -> bool:
    """Condition to check either email has subject or not"""
    email_content = step_input.previous_step_content or ""
    if email_content:
        email_content = email_content.lower()
        if "subject" in email_content:
            return True
        else:
            return False
    else:
        return False


def email_output(step_input: StepInput) -> StepOutput:
    """Returns the output of the drafting step"""
    email_content = step_input.get_step_content("Email Draft Step")

    return StepOutput(content=email_content,
                      step_name="Email Output Step",
                      executor_type="function")

# =================== agents ===================================
# email draft agent


email_draft_agent = Agent(
    id="email-draft-agent",
    name="Email Draft Agent",
    instructions=["You are an expert in drafting emails",
                  "Draft clear and professional emails based on the user's input"],
    model=model
)

# ======================== steps ==================================
# email drafting step
email_draft_step = Step(
    name="Email Draft Step",
    agent=email_draft_agent,
    description="Draft and email based on user's input prompt"
)

# email output step
email_output_step = Step(
    name="Email Output Step",
    executor=email_output,
    description="Outputs my email to end user"
)


# conditional step
review_email_step = Condition(
    evaluator=review_email_condition,
    steps=[email_output_step],
    name="Reviews the drafted email if it contains the subject line"
)


# ================================= workflow ===================================
workflow = Workflow(
    id="email-workflow",
    name="Email drafting and review workflow",
    steps=[email_draft_step, review_email_step],
    description="A workflow that drafts and email based on user and reviews it if the subject line is present or not"
)


# execute workflow
workflow.print_response(
    input="Draft and email to schedule a meeting with my technical team at 6 pm and do not give subject")
