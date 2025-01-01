from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import find_dotenv,load_dotenv
import os

# loading enviornment variable
_ = load_dotenv()

# create agent First Agent is Sentiment Analysis Agent

sentiment_agent = Agent(
    name = 'Sentiment Agent',
    role = 'Search and interpret news article',
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    tools=[GoogleSearch()],
    instructions=[
        "Find relevent news articles for each company and analyze the sentiment",
        "provide sentiment scores from 1(negative) to 10(positive) with reasoning and sources."
        "cite your sources. Be specific and provide links"
    ],
    show_tool_calls= True ,
    markdown = True
)

# second agent is Finance Agent which get data from Yahoo finance and relevent stock information of that company

finance_agent = Agent(
    name = "Finance Agent",
    role = "Get financial data and interpret trends",
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    tools = [YFinanceTools(stock_price = True, analyst_recommendations = True, company_info = True)],
    instructions=[
        "Retrive stock prices , analyst recommendations, and key financial data.",
        "Focus on trends and present the data in tabels with key insights"
    ],
    show_tool_calls= True,
    markdown= True
)

# Analyst Tool work as verfiyer agent for above two agent
analyst_agent = Agent(
    name = "Analyst Agent",
    role = "Ensure thoroughness and draw conclusions.",
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    instructions=[
        "Check outputs for accuracy and completeness",
        "Synthesize data to provide a final sentiment score (1-10) with justification."
    ],
    show_tool_calls= True,
    markdown= True
)


# create multiagent team
agent_team = Agent(
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    team = [sentiment_agent,finance_agent,analyst_agent],
    instructions=[
        "Combine the expertise of all agents to provide a cohesive well-supported detail response.",
        "Always include references and dates for all data points and sources.",
        "Present all data in structured tabels for clarity and clean presentation",
        "Explain the methodology used to arive at the sentiment scores."
    ,
    ],
    show_tool_calls= True,
    markdown= True
)

agent_team.print_response(
    "Analyze the sentiment for the following comanies during the week of december 15th-28th, 2024 : MSFT, RELIANCE.NS , NVDA \n\n"
    "1. **Sentiment Analysis** : Search for relevent news article and interpret the sentiment for each company. Provide sentiment scores in scale of 1-10. Please explain resoning and cite sources. \n\n"
    "2. **Financial Data** : Analyze stock price movement, analyst recommendation, and any notable financial data.Highlight key trend and events. Present Data in table format \n\n"
    "3. **Considilated Analysis** : Combine the insight from Sentiment Analysis and Financial Data to assign final sentiment score (1-10) for each comapnay. Provide justification for score and provide summary \
    of the most important findings \n\n"
    "Ensure your response is accurate , compresensive, include refernces to sources with pulication dates."

)
                          