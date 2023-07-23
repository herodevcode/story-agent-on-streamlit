import streamlit as st
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="📚🗺️ Story explore")
st.title('📚🗺️ Story explore')

openai_api_key = st.sidebar.text_input('OpenAI API Key')

# Defining LLM from OpenAI
def create_llm(openai_api_key):
    llm = OpenAI(
     openai_api_key=openai_api_key,
     temperature=0.8,
     model_name="text-davinci-003"
    )
    prompt = PromptTemplate(
     input_variables=["query"],
     template="You are New Friendly Story writer assistant with all world's literature and pop culture knowledge. Help users with their story writing based on thier ideas. Query: {query}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Preparing Tools for Agent
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
     name = "Web Search",
     func=search.run,
     description="A useful tool for searching the Internet to find story based on the similar ideas. Worth using for getting titles of the similar story. Use precise questions."
    )

    # Creating an Agent
    agent = initialize_agent(
     agent="zero-shot-react-description",
     tools=[search_tool],
     llm=llm,
     verbose=True,
     max_iterations=3
    )
    
    return agent

def generate_response(input_text, agent):
    response = agent(input_text)
    st.info(response['output'])

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Drop you ideas and I will find relevant story')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    elif submitted:
        agent = create_llm(openai_api_key)
        generate_response(text, agent)