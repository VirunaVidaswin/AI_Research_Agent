from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import  create_tool_calling_agent, AgentExecutor
from tools import duck_search_tool, wiki_tool



load_dotenv()

# Response template
class ResearachResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model = "gpt-4o-mini")

# prompting the llm
# response = llm.invoke("what is your name")
# print(response)

# Parsing the output with the response template
parser = PydanticOutputParser(pydantic_object=ResearachResponse)


# prompt template to the llm

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", 
     """
    You are a research assistant chatbot. Use both Wikipedia and DuckDuckGo search tools as needed.
    If a query requires broad or real-time information, prefer DuckDuckGo. For factual knowledge, use Wikipedia.
    Format responses using:
    {format_instructions}
    """),
    ("placeholder", "{chat_history}"),  # Maintains conversation history
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
    ]
    ).partial(format_instructions=parser.get_format_instructions())

tools = [duck_search_tool, wiki_tool]
agent =  create_tool_calling_agent(
    llm = llm,
    prompt=prompt,
    tools=tools
)

query = input("What can i help you research in?")
agent_executor = AgentExecutor(agent = agent, tools=tools,verbose=True)
raw_response = agent_executor.invoke({"query":query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error in stuctured output",e)

