```python
#!/usr/bin/env python3
"""
# 📁 Teacher's Assistant Strands Agent

A specialized Strands agent that is the orchestrator to utilize sub-agents and tools at its disposal to answer a user query.

## What This Example Shows

"""

from strands import Agent
from strands_tools import file_read, file_write, editor
from english_assistant import english_assistant
from language_assistant import language_assistant
from math_assistant import math_assistant
from computer_science_assistant import computer_science_assistant
from no_expertise import general_assistant

from gemini import model
import os
import base64
from strands.telemetry import StrandsTelemetry
  
# Build Basic Auth header.
LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()
 
# Configure OpenTelemetry endpoint & headers
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_BASE_URL") + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

# Configure the telemetry
# (Creates new tracer provider and sets it as global)
strands_telemetry = StrandsTelemetry().setup_otlp_exporter()

# Define a focused system prompt for file operations
TEACHER_SYSTEM_PROMPT = """
You are TeachAssist, a sophisticated educational orchestrator designed to coordinate educational support across multiple subjects. Your role is to:

1. Analyze incoming student queries and determine the most appropriate specialized agent to handle them:
   - Math Agent: For mathematical calculations, problems, and concepts
   - English Agent: For writing, grammar, literature, and composition
   - Language Agent: For translation and language-related queries
   - Computer Science Agent: For programming, algorithms, data structures, and code execution
   - General Assistant: For all other topics outside these specialized domains

2. Key Responsibilities:
   - Accurately classify student queries by subject area
   - Route requests to the appropriate specialized agent
   - Maintain context and coordinate multi-step problems
   - Ensure cohesive responses when multiple agents are needed

3. Decision Protocol:
   - If query involves calculations/numbers → Math Agent
   - If query involves writing/literature/grammar → English Agent
   - If query involves translation → Language Agent
   - If query involves programming/coding/algorithms/computer science → Computer Science Agent
   - If query is outside these specialized areas → General Assistant
   - For complex queries, coordinate multiple agents as needed

Always confirm your understanding before routing to ensure accurate assistance.
"""

TEACHER_SYSTEM_PROMPT_LLM="""
    You are an intelligent teaching assistant.
    You receive student questions and decide which specialized teacher agent is best suited to answer them.
    Use your reasoning to pick the most appropriate agent based on the question.
"""

# Create a file-focused agent with selected tools
teacher_agent = Agent(model=model, 
    system_prompt=TEACHER_SYSTEM_PROMPT_LLM,
    callback_handler=None,
    tools=[math_assistant, language_assistant, english_assistant, computer_science_assistant, general_assistant],
    trace_attributes={
        "session.id": "abc-1234", # Example session ID
        "user.id": "user-email-example@domain.com", # Example user ID
        "langfuse.tags": [
            "Agent-SDK-Example",
            "Strands-Project-Demo",
            "Observability-Tutorial"
        ]
    }
)

# Example usage
if __name__ == "__main__":
    print("\n📁 Teacher's Assistant Strands Agent 📁\n")
    print("Ask a question in any subject area, and I'll route it to the appropriate specialist.")
    print("Type 'exit' to quit.")

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() == "exit":
                print("\nGoodbye! 👋")
                break

            response = teacher_agent(
                user_input, 
            )
            
            # Extract and print only the relevant content from the specialized agent's response
            content = str(response)
            print(content)
            
        except KeyboardInterrupt:
            print("\n\nExecution interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking a different question.")

```