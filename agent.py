from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# To switch providers, change the model string:
#   Google AI Studio : "google_genai:gemini-2.0-flash"
#   OpenRouter       : set OPENAI_API_KEY + OPENAI_BASE_URL=https://openrouter.ai/api/v1
#                      then use "openai:<model-name>"
model = init_chat_model("google_genai:gemini-2.0-flash")

agent = create_deep_agent(
    model=model,
    tools=[],  # default tools: planning, virtual filesystem, shell, sub-agents, context summarization
    system_prompt=(
        "You are an expert software repository analyst. "
        "When given a repository path or URL, use your tools to clone or explore it "
        "and answer questions about its architecture, code structure, and logic. "
        "Break down complex questions into subtasks and delegate to sub-agents when needed."
    ),
)
