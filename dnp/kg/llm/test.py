# from langchain.llms import OpenAI

# llm = OpenAI(openai_api_key="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") # key no longer available after uploaded to github

# response = llm("What is Neo4j?")

# print(response)

# from langchain.prompts import PromptTemplate

# template = PromptTemplate(template="""
# You are a cockney fruit and vegetable seller.
# Your role is to assist your customer with their fruit and vegetable needs.
# Respond using cockney rhyming slang.

# Tell me about the following fruit: {fruit}
# """, input_variables=["fruit"])

from langchain.graphs import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://3.236.201.118:7687",
    username="neo4j",
    password="yields-alignment-oar"
)

r = graph.query("MATCH (m:Movie{title: 'Toy Story'}) RETURN m")
print(r)
print(graph.schema)