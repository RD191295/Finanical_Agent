from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import find_dotenv,load_dotenv
import os

# loading enviornment variable
_ = load_dotenv()

#os.environ['PHI_API_KEY'] = load_dotenv('PHI_API_KEY')
#os.environ['GROQ_API_KEY'] = load_dotenv('GROQ_API_KEY')

# creating web search agent
web_search_agent = Agent(
    name = "Web search agent",
    role = "Search the web for the information",
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    tools = [DuckDuckGo()],
    instructions=['Alway include sources'],
    show_tool_calls= True,
    markdown= True
)

## create fin analysis agent
finance_agent = Agent(
    name = "Fiannce AI agent",
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True,technical_indicators = True),
    ],
    instructions=['Use tables to display the data'],
    show_tool_calls= True,
    markdown= True
)

multi_ai_agent = Agent(
    team=[web_search_agent,finance_agent],
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response('Summarise analyst recommendation and share the latest news for NVDA', stream = True)