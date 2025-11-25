# Import RAGAS libraries
from ragas.llms import LangchainLLMWrapper
from langfuse import Langfuse
import os
from ragas.metrics import AspectCritic

#from gemini import model
config = {
    "model": "gemini-2.5-flash",  # or other model IDs
    "temperature": 0.4,
    "max_tokens": None,
    "top_p": 0.8,
}

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Choose the appropriate import based on your API:
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize with Google AI Studio
evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
    model=config["model"],
    temperature=config["temperature"],
    max_tokens=config["max_tokens"],
    top_p=config["top_p"],
))

# Google AI Studio Embeddings
'''from ragas.embeddings import GoogleEmbeddings
evaluator_embeddings = GoogleEmbeddings(
    model="models/embedding-001"  # Google's text embedding model
)'''

from langchain_google_genai import GoogleGenerativeAIEmbeddings
evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Google's text embedding model
    task_type="retrieval_document"  # Optional: specify the task type
))

# Set up the evaluator LLM (we'll use the same model as our agent)
#evaluator_llm = LangchainLLMWrapper(model)

langfuse = Langfuse(
  public_key=os.environ.get('LANGFUSE_PUBLIC_KEY'),
  secret_key=os.environ.get('LANGFUSE_SECRET_KEY'),
  host=os.environ.get("LANGFUSE_BASE_URL")
)
# Metric to check if the agent fulfills all user requests
request_completeness = AspectCritic(
    name="Request Completeness",
    llm=evaluator_llm,
    definition=(
        "Return 1 if the agent completely fulfills all the user requests with no omissions. "
        "otherwise, return 0."
    ),
)

# Tool usage effectiveness metric
tool_usage_effectiveness = AspectCritic(
    name="Tool Usage Effectiveness",
    llm=evaluator_llm,
    definition=(
        "Return 1 if the agent appropriately used available tools to fulfill the user's request "
        "(such as using retrieve for menu questions and current_time for time questions). "
        "Return 0 if the agent failed to use appropriate tools or used unnecessary tools."
    ),
)

# Tool selection appropriateness metric
tool_selection_appropriateness = AspectCritic(
    name="Tool Selection Appropriateness",
    llm=evaluator_llm,
    definition=(
        "Return 1 if the agent selected the most appropriate tools for the task. "
        "Return 0 if better tool choices were available or if unnecessary tools were selected."
    ),
)

from ragas.metrics import RubricsScore

# Define a rubric for evaluating recommendations
rubrics = {
    "score-1_description": (
        """The agent can't provide a complete answer to the question."""
    ),
    "score0_description": (
        "The agent provided an answer but didn't explain why it is correct."
    ),
    "score1_description": (
        "The agent provided a complete answer and explained why it is correct."
    ),
}

# Create the recommendations metric
recommendations = RubricsScore(rubrics=rubrics, llm=evaluator_llm, name="Recommendations")

from ragas.metrics import ContextRelevance, ResponseGroundedness 

# Context relevance measures how well the retrieved contexts address the user's query
context_relevance = ContextRelevance(llm=evaluator_llm)

# Response groundedness determines if the response is supported by the provided contexts
response_groundedness = ResponseGroundedness(llm=evaluator_llm)

metrics=[context_relevance, response_groundedness]


import pandas as pd
from datetime import datetime, timedelta
from ragas.dataset_schema import (
    SingleTurnSample,
    MultiTurnSample,
    EvaluationDataset
)
from ragas import evaluate

def extract_span_components(trace):
    """Extract user queries, agent responses, retrieved contexts 
    and tool usage from a Langfuse trace"""
    user_inputs = []
    agent_responses = []
    retrieved_contexts = []
    tool_usages = []

    # Get basic information from trace
    if hasattr(trace, 'input') and trace.input is not None:
        if isinstance(trace.input, dict) and 'args' in trace.input:
            if trace.input['args'] and len(trace.input['args']) > 0:
                user_inputs.append(str(trace.input['args'][0]))
        elif isinstance(trace.input, str):
            user_inputs.append(trace.input)
        else:
            user_inputs.append(str(trace.input))

    if hasattr(trace, 'output') and trace.output is not None:
        if isinstance(trace.output, str):
            agent_responses.append(trace.output)
        else:
            agent_responses.append(str(trace.output))

    # Try to get contexts from observations and tool usage details
    try:
        for obsID in trace.observations:
            print (f"Getting Observation {obsID}")
            observations = langfuse.api.observations.get(obsID)

            for obs in observations:
                # Extract tool usage information
                if hasattr(obs, 'name') and obs.name:
                    tool_name = str(obs.name)
                    tool_input = obs.input if hasattr(obs, 'input') and obs.input else None
                    tool_output = obs.output if hasattr(obs, 'output') and obs.output else None
                    tool_usages.append({
                        "name": tool_name,
                        "input": tool_input,
                        "output": tool_output
                    })
                    # Specifically capture retrieved contexts
                    if 'retrieve' in tool_name.lower() and tool_output:
                        retrieved_contexts.append(str(tool_output))
    except Exception as e:
        print(f"Error fetching observations: {e}")

    # Extract tool names from metadata if available
    if hasattr(trace, 'metadata') and trace.metadata:
        if 'attributes' in trace.metadata:
            attributes = trace.metadata['attributes']
            if 'agent.tools' in attributes:
                available_tools = attributes['agent.tools']
    return {
        "user_inputs": user_inputs,
        "agent_responses": agent_responses,
        "retrieved_contexts": retrieved_contexts,
        "tool_usages": tool_usages,
        "available_tools": available_tools if 'available_tools' in locals() else []
    }


def fetch_traces(batch_size=10, lookback_hours=24, tags=None):
    """Fetch traces from Langfuse based on specified criteria"""
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=lookback_hours)
    print(f"Fetching traces from {start_time} to {end_time}")
    # Fetch traces
    if tags:
        traces = langfuse.api.trace.list(
            limit=batch_size,
            tags=tags,
            from_timestamp=start_time,
            to_timestamp=end_time
        ).data
    else:
        traces = langfuse.api.trace.list(
            limit=batch_size,
            from_timestamp=start_time,
            to_timestamp=end_time
        ).data
    
    print(f"Fetched {len(traces)} traces")
    return traces

def process_traces(traces):
    """Process traces into samples for RAGAS evaluation"""
    single_turn_samples = []
    multi_turn_samples = []
    trace_sample_mapping = []
    
    for trace in traces:
        # Extract components
        components = extract_span_components(trace)
        
        # Add tool usage information to the trace for evaluation
        tool_info = ""
        if components["tool_usages"]:
            tool_info = "Tools used: " + ", ".join([t["name"] for t in components["tool_usages"] if "name" in t])
            
        # Convert to RAGAS samples
        if components["user_inputs"]:
            # For single turn with context, create a SingleTurnSample
            if components["retrieved_contexts"]:
                single_turn_samples.append(
                    SingleTurnSample(
                        user_input=components["user_inputs"][0],
                        response=components["agent_responses"][0] if components["agent_responses"] else "",
                        retrieved_contexts=components["retrieved_contexts"],
                        # Add metadata for tool evaluation
                        metadata={
                            "tool_usages": components["tool_usages"],
                            "available_tools": components["available_tools"],
                            "tool_info": tool_info
                        }
                    )
                )
                trace_sample_mapping.append({
                    "trace_id": trace.id, 
                    "type": "single_turn", 
                    "index": len(single_turn_samples)-1
                })
            
            # For regular conversation (single or multi-turn)
            else:
                messages = []
                for i in range(max(len(components["user_inputs"]), len(components["agent_responses"]))):
                    if i < len(components["user_inputs"]):
                        messages.append({"role": "user", "content": components["user_inputs"][i]})
                    if i < len(components["agent_responses"]):
                        messages.append({
                            "role": "assistant", 
                            "content": components["agent_responses"][i] + "\n\n" + tool_info
                        })
                
                multi_turn_samples.append(
                    MultiTurnSample(
                        user_input=messages,
                        metadata={
                            "tool_usages": components["tool_usages"],
                            "available_tools": components["available_tools"]
                        }
                    )
                )
                trace_sample_mapping.append({
                    "trace_id": trace.id, 
                    "type": "multi_turn", 
                    "index": len(multi_turn_samples)-1
                })
    
    return {
        "single_turn_samples": single_turn_samples,
        "multi_turn_samples": multi_turn_samples,
        "trace_sample_mapping": trace_sample_mapping
    }


def evaluate_conversation_samples(multi_turn_samples, trace_sample_mapping):
    """Evaluate conversation-based samples and push scores to Langfuse"""
    if not multi_turn_samples:
        print("No multi-turn samples to evaluate")
        return None
    
    print(f"Evaluating {len(multi_turn_samples)} multi-turn samples with conversation metrics")
    conv_dataset = EvaluationDataset(samples=multi_turn_samples)
    conv_results = evaluate(
        dataset=conv_dataset,
        metrics=[
            request_completeness, 
            recommendations,
            tool_usage_effectiveness,
            tool_selection_appropriateness
        ]
        
    )
    conv_df = conv_results.to_pandas()
    
    # Push conversation scores back to Langfuse
    for mapping in trace_sample_mapping:
        if mapping["type"] == "multi_turn":
            sample_index = mapping["index"]
            trace_id = mapping["trace_id"]
            
            if sample_index < len(conv_df):
                for metric_name in conv_df.columns:
                    if metric_name not in ['user_input']:
                        try:
                            metric_value = float(conv_df.iloc[sample_index][metric_name])
                            if pd.isna(metric_value):
                                metric_value = 0.0
                            langfuse.create_score(
                                trace_id=trace_id,
                                name=metric_name,
                                value=metric_value
                            )
                            print(f"Added score {metric_name}={metric_value} to trace {trace_id}")
                        except Exception as e:
                            print(f"Error adding conversation score: {e}")
    
    return conv_df


def save_results_to_csv(conv_df=None, output_dir="evaluation_results"):
    """Save evaluation results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    
    if conv_df is not None and not conv_df.empty:
        conv_file = os.path.join(output_dir, f"conversation_evaluation_{timestamp}.csv")
        conv_df.to_csv(conv_file, index=False)
        print(f"Conversation evaluation results saved to {conv_file}")
        results["conv_file"] = conv_file
    
    return results

def evaluate_traces(batch_size=10, lookback_hours=24, tags=None, save_csv=False):
    """Main function to fetch traces, evaluate them with RAGAS, and push scores back to Langfuse"""
    # Fetch traces from Langfuse
    traces = fetch_traces(batch_size, lookback_hours, tags)
    
    if not traces:
        print("No traces found. Exiting.")
        return
    
    # Process traces into samples
    processed_data = process_traces(traces)
    
    conv_df = evaluate_conversation_samples(
        processed_data["multi_turn_samples"], 
        processed_data["trace_sample_mapping"]
    )
    
    # Save results to CSV if requested
    if save_csv:
        save_results_to_csv(conv_df)
    
    return {
        "conversation_results": conv_df
    }

if __name__ == "__main__":
    results = evaluate_traces(
        lookback_hours=2,
        batch_size=20,
        tags=["Agent-SDK-Example"],
        save_csv=True
    )
    
    # Access results if needed for further analysis
    if results:
        if "conversation_results" in results and results["conversation_results"] is not None:
            print("\nConversation Evaluation Summary:")
            print(results["conversation_results"].describe())
