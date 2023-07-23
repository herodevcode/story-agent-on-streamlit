import streamlit as st
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

st.set_page_config(page_title="📚🗺️ Story explorer")
st.title('📚🗺️ Story explorer')

openai_api_key = st.sidebar.text_input('OpenAI API Key')

# Defining LLM from OpenAI
def create_llm(openai_api_key):
    llm = OpenAI(
     openai_api_key=openai_api_key,
     temperature=0.8,
     model_name="text-davinci-003",
     max_tokens=1000
    )

    prompt_a = PromptTemplate(
    input_variables=["ideas"],
    template= """
        instruction:
        ///
        You are an advanced story researcher AI with an expansive 
        knowledge base that covers countless works of 
        world literature and pop culture.
        Your task is to assist me 
        in reseaching about other narratives 
        similar to my ideas. 
        My ideas: {ideas}. 
        Now, let's think about what questions should I ask to google
        to get big list of most relevant stories.
    """
    )

    llm_chain_a = LLMChain(llm=llm, prompt=prompt_a)

        # Preparing Tools for Agent
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
     name = "Web Search",
     func=search.run,
     description="""A useful tool for 
     searching the Internet to find title of story 
     based on the similar ideas. 
     Worth using for getting inspiration. 
     Use precise questions and use it one for each story"""
    )

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wiki_tool = Tool(
     name = "Wikipedia",
     func=wikipedia.run,
     description="""A useful tool for going deep 
     to retrieve the information about the story"""
    )

    # Creating an Agent
    agent = initialize_agent(
     agent="zero-shot-react-description",
     tools=[search_tool, wiki_tool],
     llm=llm,
     verbose=True,
     max_iterations=5
    )
    agent_chain = agent

    prompt_b = PromptTemplate(
        input_variables=["story_list"],
        template= """
            instruction:
            ///
            You are going to read a list of stories as input and, 
            by using semantic analysis, similarity detection, and 
            multi-dimensional clustering algorithms, from different genres and time periods.
            You should identify narratives and storytelling techniques.

            Upon completion of this process, return list of story you analyzed.
            A list should includes 'Title', 'Author', 'Genre', 'Publication Year', 
            'Summary', 'Similarity Score', 'Storytelling frameworks' and 'Key takeaway from the story'. 
            ///

            Story list:
            ///
            {story_list}.
            ///
        """
        )

    llm_chain_b = LLMChain(llm=llm, prompt=prompt_b)

    overall_chain = SimpleSequentialChain(
                  chains=[llm_chain_a, agent_chain, llm_chain_b],
                  verbose=True) 
    
    return overall_chain

def generate_response(input_text, overall_chain):
    response = overall_chain(input_text)
    st.info(response['output'])

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Drop you ideas and I will find relevant story')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    elif submitted:
        agent = create_llm(openai_api_key)
        generate_response(text, agent)