import os  # Import os to interact with the operating system
import streamlit as st  # Import Streamlit for building the web app
from dotenv import load_dotenv  # Import dotenv to load environment variables
from langchain.prompts import PromptTemplate  # Import PromptTemplate to create custom prompts
from langchain_groq import ChatGroq  # Import ChatGroq for using Groq API with LangChain
from langchain.chains.summarize import load_summarize_chain  # Import summarize chain for summarization
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader  # Import document loaders
from langchain_huggingface import HuggingFaceEndpoint  # Import HuggingFaceEndpoint for using HuggingFace models
import validators

# Load environment variables from .env file
load_dotenv()

# Get the HuggingFace API Token from environment variables
hf_api_key = os.getenv("HF_TOKEN")
# Streamlit app configuration
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website")
st.markdown("""
    <style>
    .main-title {
        
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .subheader {
        
        font-size: 30px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Langchain: Summarize Text From YT or Website</h1>', unsafe_allow_html=True)
st.subheader('Summarize the URL')

# URL input field
generic_url = st.text_input("URL", label_visibility="collapsed")

# Check if API key is available
if not hf_api_key:
    st.error("HuggingFace API token is missing. Please set it in the .env file.")
else:
    # Initialize HuggingFace model
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_api_key)

    # Prompt template for summarization
    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the Content from YT or Website"):
        # Validate inputs
        if not generic_url.strip():
            st.error("Please provide a URL to summarize.")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YouTube video URL or a website URL.")
        else:
            try:
                with st.spinner("Waiting..."):
                    # Load data from the URL
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                        )
                    docs = loader.load()

                    # Chain for summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)

                    st.success(output_summary)
            except Exception as e:
                st.exception(f"Exception: {e}")
