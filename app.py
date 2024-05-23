import os
import streamlit as st

#Approach 1
from dotenv import load_dotenv
import google.generativeai as genai 

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variables
# list all env variables with os.environ
api_key = os.getenv("GOOGLE_API_KEY")
# print(api_key)

# Configure the Google Generative AI with the API key
genai.configure(api_key=api_key)

#Approach 2
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
# print(PROJECT_ID, LOCATION)

vertexai.init(project=PROJECT_ID, location=LOCATION)


@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    return text_model_pro, multimodal_model_pro

@st.cache_resource
def load_model():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.0-pro")
    return text_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)


# from IPython.display import Markdown
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
#     SystemMessagePromptTemplate,
# )
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
# from vertexai.generative_models import Content, GenerativeModel, Part

with st.sidebar:
    st.write("ChatBot")
    with st.expander("Open Chatbot"):
            # model = GenerativeModel("gemini-1.0-pro")
            # responses = model.generate_content("Why is the sky blue?", stream=True)

            # for response in responses:
            #     print(response.text, end="")
        messages = st.container(height=300)
        if prompt := st.chat_input("Say something"):
            #Question
            messages.chat_message("user").write(prompt)
            
            #Answer
            messages.chat_message("assistant").write(f"Echo: {prompt}")
    
    ## Chat example
    # chat = model.start_chat()
    # response = chat.send_message(
    #     """You are an astronomer, knowledgeable about the solar system.
    # How many moons does Mars have? Tell me some fun facts about them.
    # """
    # )
    # print(response.text)
    

# Main page
st.header("Vertex AI Gemini 1.0 API", divider="rainbow")
# text_model_pro, multimodal_model_pro = load_models()
text_model_pro = load_models()

tab1, tab2, tab3 = st.tabs(
    ["Generate Note Summary", "Patient Timeline", "Discharge"]
)

with tab1:
    st.write("Using Gemini 1.0 Pro - Text only model")
    st.subheader("Generate a summary")


    length_of_story = st.radio(
        "Select the length of the story: \n\n",
        ["Short", "Long"],
        key="length_of_story",
        horizontal=True,
    )

    max_output_tokens = 2048

    prompt = f"""Write a {length_of_story} summary based on the following notes: \n
    
    Note: <put note var here> \n
    Important point is that each chapters should be generated based on the premise given above.
    First start by giving the book introduction, chapter introductions and then each chapter. It should also have a proper ending.
    The book should have prologue and epilogue.
    """

    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Generate my story", key="generate_t2t")

    if generate_t2t and prompt:
        # st.write(prompt)
        with st.spinner("Generating your story using Gemini 1.0 Pro ..."):
            first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
            with first_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Your story:")
                    st.write(response)
            with first_tab2:
                st.text(prompt)

with tab2:
    st.write("Using Gemini 1.0 Pro - Text only model")
    st.subheader("Generate your marketing campaign")


    prompt = f"""Generate a timeline for the patient based on the notes:

    Note: <put note var here> \n

    With these inputs, make sure to follow following guidelines and generate the marketing campaign with proper headlines: \n
    - Briefly describe company, its values, mission, and target audience.
    - Highlight any relevant brand guidelines or messaging frameworks.
    - Provide a concise overview of the campaign's objectives and goals.
    - Briefly explain the product or service being promoted.
    - Define your ideal customer with clear demographics, psychographics, and behavioral insights.
    - Understand their needs, wants, motivations, and pain points.
    - Clearly articulate the desired outcomes for the campaign.
    - Use SMART goals (Specific, Measurable, Achievable, Relevant, and Time-bound) for clarity.
    - Define key performance indicators (KPIs) to track progress and success.
    - Specify the primary and secondary goals of the campaign.
    - Examples include brand awareness, lead generation, sales growth, or website traffic.
    - Clearly define what differentiates your product or service from competitors.
    - Emphasize the value proposition and unique benefits offered to the target audience.
    - Define the desired tone and personality of the campaign messaging.
    - Identify the specific channels you will use to reach your target audience.
    - Clearly state the desired action you want the audience to take.
    - Make it specific, compelling, and easy to understand.
    - Identify and analyze your key competitors in the market.
    - Understand their strengths and weaknesses, target audience, and marketing strategies.
    - Develop a differentiation strategy to stand out from the competition.
    - Define how you will track the success of the campaign.
   -  Utilize relevant KPIs to measure performance and return on investment (ROI).
   Give proper bullet points and headlines for the marketing campaign. Do not produce any empty lines.
   Be very succinct and to the point.
    """
    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Generate patient timeline", key="generate_timeline")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Timeline", "Prompt"])
        with st.spinner("Generating your marketing campaign using Gemini 1.0 Pro ..."):
            with second_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Your patient timeline:")
                    st.write(response)
            with second_tab2:
                st.text(prompt)

with tab3:
    st.write("Using Gemini 1.0 Pro - Text only model")
    st.subheader("Prepare Patient Discharge")


    prompt = f"""Generate a patient discharge based on the patient note:

    Note: <put note var here> \n

    With these inputs, make sure to follow following guidelines and generate the marketing campaign with proper headlines: \n
    - Briefly describe company, its values, mission, and target audience.
    - Highlight any relevant brand guidelines or messaging frameworks.
    - Provide a concise overview of the campaign's objectives and goals.
    - Briefly explain the product or service being promoted.
    - Define your ideal customer with clear demographics, psychographics, and behavioral insights.
    - Understand their needs, wants, motivations, and pain points.
    - Clearly articulate the desired outcomes for the campaign.
    - Use SMART goals (Specific, Measurable, Achievable, Relevant, and Time-bound) for clarity.
    - Define key performance indicators (KPIs) to track progress and success.
    - Specify the primary and secondary goals of the campaign.
    - Examples include brand awareness, lead generation, sales growth, or website traffic.
    - Clearly define what differentiates your product or service from competitors.
    - Emphasize the value proposition and unique benefits offered to the target audience.
    - Define the desired tone and personality of the campaign messaging.
    - Identify the specific channels you will use to reach your target audience.
    - Clearly state the desired action you want the audience to take.
    - Make it specific, compelling, and easy to understand.
    - Identify and analyze your key competitors in the market.
    - Understand their strengths and weaknesses, target audience, and marketing strategies.
    - Develop a differentiation strategy to stand out from the competition.
    - Define how you will track the success of the campaign.
   -  Utilize relevant KPIs to measure performance and return on investment (ROI).
   Give proper bullet points and headlines for the marketing campaign. Do not produce any empty lines.
   Be very succinct and to the point.
    """
    config = {
        "temperature": 0.8,
        "max_output_tokens": 2048,
    }

    generate_t2t = st.button("Generate patient discharge", key="generate_discharge")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Discharge", "Prompt"])
        with st.spinner("Generating your marketing campaign using Gemini 1.0 Pro ..."):
            with second_tab1:
                response = get_gemini_pro_text_response(
                    text_model_pro,
                    prompt,
                    generation_config=config,
                )
                if response:
                    st.write("Your patient timeline:")
                    st.write(response)
            with second_tab2:
                st.text(prompt)
   