import streamlit as st
import streamlit.components.v1 as components
from langchain_community.chat_message_histories import ChatMessageHistory

from core import GitHubQA

# GCP LLM Configurations using Google API key and goog.egenerativeai
# --------------------
import google.generativeai as genai
GOOGLE_API_KEY = "AIzaSyATIUK_dvY0rg_FONGW1668--IwlMrFVyo"
GOOGLE_PROJECT = "analog-grin-455718-i4"
GOOGLE_LOCATION = "us-central1"
genai.configure(api_key=GOOGLE_API_KEY)
# --------------------

# GCP LLM Configurations using local credentails works on GCP cloud run with google genai and google.auth
# No need to store public key in code
# --------------------
# from google import genai
# import google.auth

# credentials, project = google.auth.default()

# client = genai.Client(
#     credentials=credentials,
#     project="analog-grin-455718-i4"
# )
# --------------------

model_name = "gemini-1.5-pro-002"

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Main background and text colors */
        body {
            color: white;
            background-color: #0E1117;
        }
        
        .stApp {
            background-color: #0E1117;
            color: white;
            overflow: visible !important;
        }
        
        /* Header styling with custom green */
        h1, h2, h3, h4, h5, h6 {
            color: #86BC25 !important; /* Vibrant green for headers */
        }
        
        /* Button styling */
        .stButton button {
            background-color: #86BC25;
            color: black;
            border: none;
            border-radius: 5px;
        }
        
        /* Input field styling */
        .stTextInput input, .stSelectbox, .stFileUploader {
            background-color: #1A1E29;
            color: white;
            border: 1px solid #86BC25;
            border-radius: 5px;
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: #1A1E29;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }
        
        .stChatMessage.user {
            background-color: #2C3339;
            border-left: 3px solid #86BC25;
        }
        
        .stChatMessage.assistant {
            background-color: #1A1E29;
            border-left: 3px solid #86BC25;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #171B26;
        }
        
        /* Custom logo container */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        
        /* Dialogflow messenger custom styling */
        df-messenger {
            position: fixed !important;
            bottom: 20px !important;
            right: 20px !important;
            z-index: 9999 !important;
            /* theme overrides */
            --df-messenger-font-color: white;
            --df-messenger-font-family: 'Google Sans', sans-serif;
            --df-messenger-chat-background: #1A1E29;
            --df-messenger-message-user-background: #86BC25;
            --df-messenger-message-user-font-color: black;
            --df-messenger-message-bot-background: #2C3339;
            --df-messenger-chat-min-height: 430px;
            --df-messenger-chat-max-height: 80vh;
            --df-messenger-chat-min-width: 300px;
            --df-messenger-chat-max-width: 400px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_brand_logo():
    st.markdown(
        """
        <div class="logo-container">
            <img src="https://logos-world.net/wp-content/uploads/2021/08/Deloitte-Logo.png" width="200">
        </div>
        """, 
        unsafe_allow_html=True
    )
    
def render_dialogflow_messenger():
    html = """
    <link rel="stylesheet"
          href="https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/themes/df-messenger-default.css">
    <script src="https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/df-messenger.js"></script>

    <df-messenger
      project-id="analog-grin-455718-i4"
      agent-id="40243f29-c390-4320-88b3-af14a8a24a82"
      language-code="en"
      chat-title="Your WayFinder Assistant">
    </df-messenger>
    """
    components.html(html, height=600, width=400)
    
question_list = [
    "Select this to enter your custom question",
    "How can I extend this class to include multiplication and division methods?",
    "Please provide test cases for the addition class.",
    "Please provide test cases for different functions in this code.",
    "Please provide the scope for this document.",
    "Please provide a brief about this project for the HR manager highlighting the skills needed for a developer who will extend this project.",
    "Please suggest edits in code formatting to support OOP disciplines."
]
    
def create_prompt(code, question):
    return f"""You are a knowledgeable software assistant. Analyze the following document and answer the subsequent question.
        [CODE START]
        {code}
        [CODE END]
        Question: {question}
        Please provide a clear and concise answer. If you generate code, make sure it is runnable given the code above (Don't include placeholders).
    """

# Streamlit page setup
st.set_page_config(
    page_title="🤖 Your WayFinder AI Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# Display logo at the top
render_brand_logo()

if 'show_chat_assistant_enabled' not in st.session_state:
    st.session_state.show_chat_assistant_enabled = False
if 'show_repo_assistant_enabled' not in st.session_state:
    st.session_state.show_repo_assistant_enabled = False

def toggle_chat_assistant():
    if st.session_state.show_chat_assistant_key:
        st.session_state.show_chat_assistant_enabled = True
        st.session_state.show_repo_assistant_enabled = False
    else:
        st.session_state.show_chat_assistant_enabled = False

# Function to handle checkbox 2 change
def toggle_repo_assistant():
    if st.session_state.show_repo_assistant_key:
        st.session_state.show_repo_assistant_enabled = True
        st.session_state.show_chat_assistant_enabled = False
    else:
        st.session_state.show_repo_assistant_enabled = False

# Create a header with branded content
st.markdown("<h1 style='text-align: center;'>Your WayFinder Code Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A3A8B8;'>Powered by GCP to help you with code analysis and development</p>", unsafe_allow_html=True)

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chatMessageHistory = ChatMessageHistory()

# Top‑level checkbox to switch modes
show_chat_assistant = st.sidebar.checkbox("Show General Chat Assistant",
                                          key="show_chat_assistant_key",
                                          value=st.session_state.show_chat_assistant_enabled,
                                          disabled=st.session_state.show_repo_assistant_enabled,
                                          on_change=toggle_chat_assistant
                                          )

# if show_chat_assistant != st.session_state.show_chat_assistant_enabled:
#     st.session_state.show_chat_assistant_enabled = show_chat_assistant

# Add branding to sidebar
st.sidebar.markdown("## WayFinder Chat")
st.sidebar.markdown("Chat Assistant Agent")
# st.sidebar.markdown("Check the box above to show the project chat assistant.")

# Add a divider
st.sidebar.markdown("---")

show_repo_assistant = st.sidebar.checkbox("Show Github Repo Assistant",
                                          key="show_repo_assistant_key",
                                          value=st.session_state.show_repo_assistant_enabled,
                                          disabled=st.session_state.show_chat_assistant_enabled,
                                          on_change=toggle_repo_assistant
                                          )

# if show_repo_assistant != st.session_state.show_chat_assistant_enabled:
#     st.session_state.show_chat_assistant_enabled = show_repo_assistant

st.sidebar.markdown("## WayFinder Repo Assistant")
st.sidebar.markdown("Github Repo Agent")
# st.sidebar.markdown("Check the box above to show the project chat assistant.")


# Add company info to sidebar
st.sidebar.markdown("### WayFinder")
st.sidebar.markdown("Deloitte WayFinders powered by GCP AI technologies to help developers get answers about project history and to guide them on their coding tasks.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #A3A8B8; padding: 20px;">
        <p>© 2025 Deloitte WayFinders. All rights reserved.</p>
        <p>Built with GCP technologies</p>
    </div>
    """,
    unsafe_allow_html=True
)

if show_chat_assistant:
    st.subheader("Project Chat Assistant")
    st.write("Use the Chat Assistant to ask questions about the project.")
    st.sidebar.success("Assistant Chat Enabled")
    render_dialogflow_messenger()

elif show_repo_assistant:
    st.subheader("Project Repo Assistant")
    st.write("Use the Chat Assistant to ask questions about the project code.")
    st.sidebar.success("Github Repo Agent Enabled")
    
    github_repo = st.text_input("Github Repo URL", placeholder="github/codespaces-jupyter")

    if github_repo == "":
        st.info("Please insert Github Repo.")
    
    repo_agent_container = st.container()
    
    with repo_agent_container:
        st.markdown("---")
        st.subheader("💬 Chat with Your Repo Agent")
        
        # Display all past messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_input = None
            
        if prompt := st.chat_input("Ask any question about your code..."):
            if github_repo == "":
                with st.chat_message("assistant"):
                    st.warning("Please insert Github Repo before asking questions.")
            else:
                user_input = prompt
                
        if user_input and github_repo:
            st.session_state.chatMessageHistory.add_message({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "user", "content": user_input})
                
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Add a loading indicator
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your code..."):
                    try:
                        
                        qa = GitHubQA(repo=github_repo)
                        if qa.qa: # Check if QA chain was successfully initialized
                            ans, srcs = qa.get_answer(user_input)
                        else:
                            st.error("QA system failed to initialize. Skipping question.")
                        
                        if ans:
                            st.session_state.chatMessageHistory.add_message({"role": "assistant", "content": ans})
                            st.session_state.messages.append({"role": "assistant", "content": ans})
                            st.markdown(ans)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Details: " + str(type(e).__name__))
                    
else:
    col1, col2 = st.columns(2)
    
    # st.subheader("Code Analysis")
    # st.write("Upload your code file and type a question to analyze it using our advanced AI model:")
    
    with col1:
        st.subheader("Upload a file")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "py"])
        
        if uploaded_file is not None:
            code = uploaded_file.read().decode("utf-8")
            with st.expander("View uploaded code", expanded=False):
                st.code(code, language="python")
        else:
            code = ""
            st.info("Please upload a code file to analyze")
            
        # Add a "Request Analysis" button for better UX
        # if st.button("🔍 Analyze Code", disabled=(uploaded_file is None)):
        #     st.session_state.analyze_clicked = True
        
    with col2:
        st.subheader("Ask Questions")
        question = st.selectbox("Choose a question to ask:", question_list)
        
    # Create a container for the chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display a divider
        st.markdown("---")
        st.subheader("💬 Chat with Your AI Assistant")
        
        # Display all past messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            
    user_input = None
    if question and question != "Select this to enter your custom question" and uploaded_file is not None:
        full_prompt = create_prompt(code, question)
        user_input = question
        
    elif prompt := st.chat_input("Ask any question about your code..."):
        if uploaded_file is None:
            with st.chat_message("assistant"):
                st.warning("Please upload a code file first before asking questions.")
        else:
            full_prompt = create_prompt(code, prompt)
            user_input = prompt
            
    if user_input and uploaded_file is not None:
        st.session_state.chatMessageHistory.add_message({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "user", "content": user_input})
            
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add a loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your code..."):
                try:
                    # Using gogle.generativeai package
                    # --------------------
                    client = genai.GenerativeModel(model_name)
                    response = client.generate_content(full_prompt)
                    answer = response.text.strip()
                    # --------------------
                    
                    # Using google.genai
                    # --------------------
                    # response = client.models.generate_content(
                    #     model=model_name,
                    #     contents=[full_prompt],
                    # )
                    # answer = response.text
                    # --------------------
                    
                    if answer:
                        st.session_state.chatMessageHistory.add_message({"role": "assistant", "content": answer})
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.markdown(answer)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Details: " + str(type(e).__name__))