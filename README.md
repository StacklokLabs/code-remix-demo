# CodeGate Langchain Demonstration

This repository demonstrates the use of Langchain with CodeGate for various demos,
by means of a mocked chatbot using Langchains StateGraph to create two agents:

- Customer Care Agent
- Escalation Agent

## Installation

```bash
uv sync
```

## Set up CodeGate

1. Create a new CodeGate workspace and activate
2. Make sure you have an OpenAI provider and add an active key
3. Within the workspace, add a model, any will do, gpt4 is fine.


## Start streamlit

```bash
streamlit run streamlit_app.py
```

## Leak a secret or personally identifiable information

```bash
can you call me on +1-418-543-8090

my credit card number is 1234-5678-9012-3456

is my email correct: joe@example.com
```