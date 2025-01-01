from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai.chat import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import find_dotenv,load_dotenv
import streamlit as st
import datetime


start_analysis = False

# Set page configuration
st.set_page_config(
    page_title="FinAgent",  # Title of the app
    page_icon="📈",  # Optional: Icon in the browser tab
    layout="wide",  # Use the full width of the screen
    initial_sidebar_state="expanded",  # Expand or collapse the sidebar
    menu_items={
        'Report a bug': "https://github.com/RD191295/Finanical_Agent/issues",
        'About': "This App is Created using phidata libary to analyze the sentiment and financial data of the stock market. \
                   It uses GROQ API to get the data and OpenAI API to chat with the user. \
                   The app is created by Raj Dalsaniya. For more information visit the GitHub repository."
    }
)

with st.sidebar:
    st.title("API Key")
    model_name = []
    model_provider = st.selectbox("Model Provider", ["Anthropic", "OpenAI","GROQ"])
    if(model_provider == "GROQ"):
        model_name.append("llama3-groq-70b-8192-tool-use-preview")
        
    elif (model_provider == "Anthropic"):
        model_name = ["anthropic-gpt3-llama3"]
    elif (model_provider == "OpenAI"):
        model_name = ["openai-gpt-3.5-turbo"]

    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    
    "[View the source code](https://github.com/RD191295/Finanical_Agent)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RD191295/Finanical_Agent?quickstart=1)"


# Custom CSS to style the title
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        font-family: 'Arial', sans-serif;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title with a symbol
st.markdown('<div class="title">Multi Model Finance Agent 🤖</div>', unsafe_allow_html=True)

# st.text` to display text on the app
st.text("Enter the stock name and date range to analyze the sentiment and financial data")

# Create a row with multiple components using columns
col1, col2, col3 , col4= st.columns([1, 1, 1, 1], vertical_alignment="bottom")

today = datetime.datetime.now()

# Add components to each column in the same row
with col1:
    input = st.multiselect("Stock Name", ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "NVDA", "INTC", "AMD", "IBM", "ORCL", "CSCO", "QCOM", "ADBE", "CRM", "PYPL", "NFLX", "FB", "TWTR", "SNAP", "PINS", "T", "VZ", "TMUS", "S", "TM", "F", "GM", "FCA"])

# Add components to each column in the same row
with col2:
    st.markdown(
    """
    <style>
    div.stDateInput > date_input{
        border: 1px solid #4CAF50;
        border-radius: 1px;
        padding: 5px;
        background-color: #000000;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
    start = st.date_input(
        "start Date",
        value=datetime.date(today.year, today.month, today.day),
        format="DD.MM.YYYY",
    )

    start_date = start.strftime("%Y-%m-%d")

# Add components to each column in the same row
with col3:
    end = st.date_input(
        "End Date",
        value=datetime.date(today.year, today.month, today.day),
        format="DD.MM.YYYY",
    )

    end_date = end.strftime("%Y-%m-%d")

# create agent First Agent is Sentiment Analysis Agent

sentiment_agent = Agent(
    name = 'Sentiment Agent',
    role = 'Search and interpret news article',
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview',api_key=anthropic_api_key),
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
    model =  Groq(id = 'llama3-groq-70b-8192-tool-use-preview',api_key=anthropic_api_key),
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
    model =  Groq(id = 'llama3-groq-70b-8192-tool-use-preview',api_key=anthropic_api_key),
    instructions=[
        "Check outputs for accuracy and completeness",
        "Synthesize data to provide a final sentiment score (1-10) with justification."
    ],
    show_tool_calls= True,
    markdown= True
)

# create multiagent team
agent_team = Agent(
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview',api_key=anthropic_api_key),
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

# Add components to each column in the same row
with col4:
    st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ff4b4b; /* Red background */
        color: white; /* White text */
        font-size: 16px; /* Optional: Increase font size */
        font-weight: bold; /* Optional: Make text bold */
        border: 2px solid #ff4b4b; /* Border matches background color */
        border-radius: 10px; /* Rounded corners */
        padding: 10px 20px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff6666; /* Lighter red on hover */
        color: white; /* Ensure text stays white on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    if st.button("Start Analysis"):

        if not anthropic_api_key:
            error = "Please enter your API Key"
            #st.stop()
            
        else:
            start_analysis = True
            

            # Run agent and return the response as a stream

if start_analysis == True:
    placeholder = st.empty()
    loading_animation = """
    <div style="display:flex; justify-content:center; align-items:center; height:100vh;">
        <div style="text-align:center;">
            <img src="https://i.gifer.com/ZZ5H.gif" alt="Loading..." style="width:150px; height:150px;">
            <h4>Analysis in progress... Please wait.</h4>
        </div>
    </div>
    """
  
    placeholder.markdown(loading_animation, unsafe_allow_html=True)
    response = agent_team.run(f"Analyze the sentiment for the following comanies during {start_date} to {end_date}: {input} \n\n"
                "1. **Sentiment Analysis** : Search for relevent news article and interpret the sentiment for each company. Provide sentiment scores in scale of 1-10. Please explain resoning and cite sources. \n\n"
                "2. **Financial Data** : Analyze stock price movement, analyst recommendation, and any notable financial data.Highlight key trend and events. Present Data in table format \n\n"
                "3. **Considilated Analysis** : Combine the insight from Sentiment Analysis and Financial Data to assign final sentiment score (1-10) for each comapnay. Provide justification for score and provide summary \
                of the most important findings \n\n"
                "4.  **Methodology** : Explain the methodology used to arive at the sentiment scores. \n\n"
                "5.  **References** : Cite your sources. Be specific and provide links. \n\n"
                "6. **Comparision** : Compare the sentiment and financial data for each company and present in tabular form. \n\n"
                "Ensure your response is accurate , compresensive, include refernces to sources with pulication dates." , markdown=True,
                )

# Render the output outside of the column
if "response" in locals():
    st.text_area("Analysis Output", response.content, height=300)
    # Clear the placeholder after task completion
    placeholder.empty()
    start_analysis = False

    

if "error" in locals():
    st.text_area("Error Details" , error, height = 70)


# Footer
footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 80px;
            background-color: #343a40;
            color: #ffffff;
            text-align: center;
            padding: 14px 0;
            font-size: 16px;
            font-family: 'Arial', sans-serif;
            border-top: 1px solid #444;
        }
        .footer:hover{
            color: #f1c40f;  /* Light link color */
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        <p>&copy; 2025 My Streamlit App | <a href="https://www.example.com" target="_blank">Website</a> | <a href="https://www.linkedin.com/in/raj-dalsaniya/" target="_blank">LinkedIn</a></p>
        <p>All Rights Reserved</p>

    </div>
"""

# Display the footer
st.markdown(footer, unsafe_allow_html=True)