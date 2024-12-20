from st_helper import *

###############################################################################

llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                        openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature = 0)

###############################################################################

# Set web page title, icon, and layout
st.set_page_config(
    page_title="ESIA Analyzer 💬 WSP",
    page_icon=":robot:",
    layout="wide"
)

# Sidebar
st.sidebar.image("https://download.logo.wine/logo/WSP_Global/WSP_Global-Logo.wine.png", width=100)
st.sidebar.markdown("#")
st.sidebar.markdown("Chat History")
st.sidebar.markdown("- Describe a project for which an ESIA was done in this country.")

if "typewriter_executed" not in st.session_state:
    st.session_state.typewriter_executed = False

header = "Hi, how can I help you today?"
subheader = "Dive into the world of environmental impact analysis with your AI guide, powered by WSP Digital Innovation!"

if not st.session_state.typewriter_executed:
    speed = 10
    typewriter_header(text=header, speed=speed)
    speed = 10
    typewriter_subheader(text=subheader, speed=speed)
    st.session_state.typewriter_executed = True
    
else:
    st.markdown(f"<h1 style='color: #F9423A; text-align: center;'>{header}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: #F9423A; text-align: center;'>{subheader}</h4>", unsafe_allow_html=True)

###############################################################################

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Data
countries = ["Spain", "Romania", "Finland", "Germany", "Greece", "Turkey", "Austria", "Turkey", 
             "Belgium", "Cyprus", "France", "Sweden", "Italy", "Denmark"]
latitudes = [40.4637, 45.9432, 61.9241, 51.1657, 39.0742, 39.9208, 47.5162, 39.9208, 
             50.8503, 35.1264, 46.6034, 60.1282, 41.8719, 56.2639]
longitudes = [-3.7492, 24.9668, 25.7482, 10.4515, 21.8243, 32.8597, 14.5501, 32.8597, 
              4.3517, 33.4299, 1.8883, 18.6435, 12.5674, 9.5018]

# Create DataFrame
df_cities = pd.DataFrame({
    'country': countries,
    'lat': latitudes,
    'lon': longitudes
})

# Initialize session state for selected countries
if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = []

# Multiselect widget to filter countries
selected_index = st.multiselect(
    "Use this filter to search for information on specific countries:",
    countries,
    default=st.session_state.selected_countries  # Prepopulate with saved selection
)

# Update session state
if selected_index != st.session_state.selected_countries:
    st.session_state.selected_countries = selected_index

# Filter DataFrame based on selection
if selected_index:
    df_cities = df_cities[df_cities['country'].isin(selected_index)]

# Display map
st.map(df_cities)
    
st.markdown('#') 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# RAG

if init_prompt := st.chat_input("Ask anything"):
    
    st.chat_message("user").markdown(init_prompt)
    
    st.session_state.messages.append({"role": "user", "content": init_prompt})
    
    # Parameters
    indexes = [index_name]
    top_k = 20
    rr_th = 0
    
    # Answer generation
    question = str(init_prompt)
    
    prompt = AGENT_DOCSEARCH_PROMPT
    
    tools = [GetDocSearchResults_Tool(
    indexes=indexes, k=top_k, filters= "", reranker_th=rr_th, sas_token='na')]

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
    )
    
    session_id = 123
    
    with st.spinner("Retrieving results..."):
        response = with_message_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )

    history = update_history(session_id, question, response["output"], indexes)

    history = history[-3:]
    
    full_response = {
        "question": question,
        "output": response["output"],
        "history": history
    }

    response_text = full_response['output']   
    response = f"{response_text}"
    
    # Display answer
    with st.chat_message("assistant"):
        
        st.write(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})