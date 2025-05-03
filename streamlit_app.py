import streamlit as st
import re # Import regex module
from graph_logic import compiled_app # Import the compiled LangGraph app

st.header("⚡ Sparky Electric Customer Support")
st.subheader("💬 Chat with our support team")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display escalation summary if it exists for this agent turn
        if "escalation" in message:
            with st.expander("See Internal Escalation Summary"):
                st.markdown(message["escalation"])

# Accept user input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the input for the LangGraph app
    inputs = {"user_question": prompt}

    # Invoke the LangGraph app
    # Use st.spinner for user feedback during processing
    with st.spinner('Typing...'):
        try:
            result = compiled_app.invoke(inputs)
            support_response = result.get("support_agent_notes", "Sorry, I couldn't generate a response.")
            escalation_summary = result.get("escalation_summary", "No escalation summary generated.")
            redaction_occurred = result.get("redaction_occurred", False)
            protection_message = result.get("protection_message", None)

            # --- Replace UUID pattern --- >
            uuid_pattern = re.compile(r'<[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}>')
            replacement_text = "[Reference ID]"
            support_response = uuid_pattern.sub(replacement_text, support_response)
            escalation_summary = uuid_pattern.sub(replacement_text, escalation_summary)
            # <----------------------------

            # Prepare the agent's response message content
            agent_response_content = ""
            if redaction_occurred and protection_message:
                # Prepend the CodeGate protection message if redaction happened
                agent_response_content += f"*Notice: {protection_message}*\n\n"
            agent_response_content += support_response

            # Add agent response to chat history and display it
            agent_message = {
                "role": "assistant",
                "content": agent_response_content,
                "escalation": escalation_summary # Store escalation summary with the message
            }
            st.session_state.messages.append(agent_message)

            with st.chat_message("assistant"):
                st.markdown(agent_response_content)
                # Display the escalation summary in an expander
                with st.expander("See Internal Escalation Summary"):
                    st.markdown(escalation_summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Add error message to chat history
            error_message = {"role": "assistant", "content": f"Sorry, an error occurred processing your request: {e}"}
            st.session_state.messages.append(error_message) 