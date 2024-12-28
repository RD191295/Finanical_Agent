from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.python import PythonTools
from dotenv import find_dotenv,load_dotenv
import os

import phi
from phi.playground import Playground,serve_playground_app
# loading enviornment variable
_ = load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

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
                      company_news=True),
    ],
    instructions=['Use tables to display the data'],
    show_tool_calls= True,
    markdown= True
)

app = Playground(
    agents = [finance_agent,web_search_agent]
).get_app()


if __name__ == '__main__':
    serve_playground_app('playground:app',reload=True)
