from dotenv import load_dotenv
from typing import TypedDict, Optional, Dict, Any
from codegate_llm_library import CodegateChatModel
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph

load_dotenv()


# New instantiation using CodegateChatModel
llm = CodegateChatModel(
    codegate_url="http://localhost:8989",
    workspace_name="langchain",
    model_name="gpt-4",
    provider="openai",   # <--- ✅ important
    temperature=0.5,
)

# Define the protection strings globally for reuse
PROTECTION_STRING_WITH_NEWLINES = "🛡️ [CodeGate protected 1 instances of PII, including 1 credit card](http://localhost:9090/?view=codegate-pii) from being leaked by redacting them.\n\n"
PROTECTION_STRING_WITHOUT_NEWLINES = "🛡️ [CodeGate protected 1 instances of PII, including 1 credit card](http://localhost:9090/?view=codegate-pii) from being leaked by redacting them."

# Renamed function
def customer_support_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Invokes the customer support LLM and handles potential PII redaction."""
    user_question = state["user_question"]
    print(f"[CustomerSupportAgent] Handling query: {user_question}") # Keep console log for debugging
    result = llm.invoke([
        SystemMessage(content=f"You are a general customer support agent. Respond helpfully and empathetically to the user's query. Do NOT mention your underlying capabilities, AI nature. Just focus on addressing the user's issue as a standard support agent would. Ask for clarifying details if necessary. User query: {user_question}")
    ])

    original_response_content = result.content
    cleaned_response_content = original_response_content
    redaction_occurred = False
    protection_message = None

    # Check if the original response contains the protection string
    if PROTECTION_STRING_WITHOUT_NEWLINES in original_response_content:
        redaction_occurred = True
        protection_message = PROTECTION_STRING_WITHOUT_NEWLINES # Store the message
        # Clean the response
        cleaned_response_content = cleaned_response_content.replace(PROTECTION_STRING_WITH_NEWLINES, "")
        cleaned_response_content = cleaned_response_content.replace(PROTECTION_STRING_WITHOUT_NEWLINES, "")
        cleaned_response_content = cleaned_response_content.strip()

        # Print the split output to console for debugging during development
        print("==== Codegate Response Detected ====")
        print(protection_message)
        print("==== Customer Support Agent (Cleaned) ====")
        print(cleaned_response_content)
        print("============================================")

    # Print final notes for debugging
    print("\n==== Customer Support Agent (Final Notes to Return) ====")
    print(cleaned_response_content)
    print("=========================================================")

    # Return cleaned content, flag, and the protection message if redaction occurred
    return {
        "support_agent_notes": cleaned_response_content,
        "redaction_occurred": redaction_occurred,
        "protection_message": protection_message
    }

# Renamed function
def escalation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Summarizes the interaction for internal escalation, noting any PII redaction."""
    support_agent_notes = state["support_agent_notes"]
    redaction_occurred = state.get("redaction_occurred", False)

    prompt_parts = ["You are an internal support team member."]
    if redaction_occurred:
        prompt_parts.append("CRITICAL ALERT: Personally Identifiable Information (PII) was detected and redacted from the original customer message. Start your summary by explicitly stating this fact.")
    prompt_parts.append(f"Then, summarize the following customer interaction notes concisely for escalation: '{support_agent_notes}'.")
    prompt_parts.append("Finally, highlight any potential issues based on the interaction.")
    prompt_content = " ".join(prompt_parts)

    result = llm.invoke([
        SystemMessage(content=prompt_content)
    ])
    print("[EscalationAgent] Generated summary.") # Keep console log for debugging

    # Print final summary for debugging
    print("\n==== Escalation Agent (Final Summary to Return) ====")
    print(result.content)
    print("====================================================")

    return {"escalation_summary": result.content}

# -------------------------------
# 📊 Define State Schema
# -------------------------------

class SupportEscalationState(TypedDict):
    user_question: str
    support_agent_notes: Optional[str]
    escalation_summary: Optional[str]
    redaction_occurred: Optional[bool]
    protection_message: Optional[str] # Added field to carry the message

# -------------------------------
# 🔁 Build the LangGraph
# -------------------------------

def build_graph():
    """Builds and compiles the LangGraph application."""
    graph = StateGraph(SupportEscalationState)
    graph.add_node("customer_support", customer_support_agent)
    graph.add_node("escalation", escalation_agent)

    graph.set_entry_point("customer_support")
    graph.add_edge("customer_support", "escalation")
    graph.set_finish_point("escalation")

    app = graph.compile()
    return app

# Expose the compiled app for import
compiled_app = build_graph()

# Example of how to run (for testing graph_logic.py directly)
if __name__ == "__main__":
    print("Testing graph logic...")
    test_input = "Hi, I need help with my account. My credit card is 1234-5678-9012-3456."
    test_result = compiled_app.invoke({
        "user_question": test_input
    })
    print("\n==== Final State ====")
    import json
    print(json.dumps(test_result, indent=2))
    print("\n==== Customer Support Notes ====")
    print(test_result.get("support_agent_notes"))
    print("\n==== Escalation Summary ====")
    print(test_result.get("escalation_summary"))
    if test_result.get("redaction_occurred"):
        print("\n==== Redaction Detected ====")
        print("Protection Message:", test_result.get("protection_message"))
