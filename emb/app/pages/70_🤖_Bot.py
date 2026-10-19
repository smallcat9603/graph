"""Chat over the loaded graph: natural language -> Cypher -> answer.

dnp/kg wrapped GraphCypherQAChain in a LangChain conversational agent (removed in
LangChain 1.x). This page calls the chain directly, so each question is answered
on its own, without the 5-turn chat memory.
"""

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

from ui import core

core.require_data()
st.title(f"{st.session_state['data']} Robot")

form = st.form('my_form')
openai_api_key = form.text_input('OpenAI API Key', type='password')

if form.form_submit_button('Submit'):
    if not openai_api_key.startswith('sk-'):
        form.warning('Please enter your OpenAI API key!', icon='⚠')
        st.stop()
elif openai_api_key == "":
    st.stop()

llm = ChatOpenAI(
    api_key=openai_api_key,
    model=st.secrets["OPENAI_MODEL"],
)

graph = Neo4jGraph(
    url=st.session_state["host"],
    username=st.session_state["user"],
    password=st.session_state["password"],
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about node similarity and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Example Cypher Statements:

1. How to find the most similar node to Query node "C-1":
```
MATCH (q:Query)-[r:SIMILAR_JACCARD]-(a:Article) WHERE q.name = "C-1"
RETURN q.name AS Query, a.name AS Article, a.url AS URL, r.score AS Similarity
ORDER BY Similarity DESC
```

Schema:
{schema}

Question:
{question}
"""

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE),
    allow_dangerous_requests=True, # the LLM writes Cypher that runs on this database
)


def write_message(role, content, save=True):
    """Save a message to the session state and write it to the UI."""
    if save:
        st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Chatbot!  How can I help you?"},
    ]

for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

if prompt := st.chat_input("What is up?"):
    write_message('user', prompt)
    with st.spinner('Thinking...'):
        response = cypher_qa.invoke({"query": prompt})
        write_message('assistant', response["result"])
