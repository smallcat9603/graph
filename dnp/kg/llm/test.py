from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="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")

response = llm("What is Neo4j?")

print(response)