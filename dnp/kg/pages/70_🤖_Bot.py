import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

if 'data' not in st.session_state:
   st.title("No Graph Data")
   st.warning("You should load graph data first!", icon='⚠')
   st.stop()
else:
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
    openai_api_key=openai_api_key,
    model=st.secrets["OPENAI_MODEL"],
)

# embeddings = OpenAIEmbeddings(
#     openai_api_key=st.secrets["OPENAI_API_KEY"]
# )

# neo4jvector = Neo4jVector.from_existing_index(
#     embeddings,                              
#     url=st.secrets["NEO4J_URI"],             
#     username=st.secrets["NEO4J_USERNAME"],   
#     password=st.secrets["NEO4J_PASSWORD"],   
#     index_name="moviePlots",                 
#     node_label="Movie",                      
#     text_node_property="plot",               
#     embedding_node_property="plotEmbedding", 
#     retrieval_query="""
#     RETURN
#     node.plot AS text,
#     score,
#     {
#         title: node.title,
#         directors: [ (person)-[:DIRECTED]->(node) | person.name ],
#         actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
#         tmdbId: node.tmdbId,
#         source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
#     } AS metadata
#     """
# )

# retriever = neo4jvector.as_retriever()

# kg_qa = RetrievalQA.from_chain_type(
#     llm,                  
#     chain_type="stuff",   
#     retriever=retriever,  
# )


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

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,          
    graph=graph,  
    verbose=True,
    cypher_prompt=cypher_prompt,
)

tools = [
    # Tool.from_function(
    #     name="Vector Search Index",  
    #     description="Provides information about movie plots using Vector Search", 
    #     func = kg_qa, 
    # ),
    # Tool.from_function(
    #     name="Graph Cypher QA Chain",  
    #     description="Provides information about Movies including their Actors, Directors and User reviews", 
    #     func = cypher_qa, 
    # ),
    Tool.from_function(
    name="Graph Cypher QA Chain",  
    description="Provides information", 
    func = cypher_qa, 
    ),
]

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

SYSTEM_MESSAGE = """
You are a expert providing information about node similariy.
Be as helpful as possible and return as much information as possible.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
"""

agent = initialize_agent(
    tools,
    llm,
    memory=memory,
    verbose=True,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    agent_kwargs={"system_message": SYSTEM_MESSAGE}
)

def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent(prompt)

    return response['output']


def write_message(role, content, save=True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        st.markdown(content)

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Chatbot!  How can I help you?"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # # # TODO: Replace this with a call to your LLM
        # from time import sleep
        # sleep(1)
        # write_message('assistant', message)

        response = generate_response(message)
        write_message('assistant', response)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
