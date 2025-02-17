import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
load_dotenv()
       
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Summarize Any URL (YouTube or Website)"

# Streamlit App Configuration
st.set_page_config(page_title="Summarize", page_icon="▶️🌐", layout="wide")
st.title("Summarize Text from YT▶️ or Website🌐")
st.subheader("Summarize Any URL (YouTube or Website)")

# Sidebar Configuration
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Groq API Key", value="", type="password")
    model_name = st.selectbox("Select LLM Model:", ["Gemma2-9b-It", "Llama3-8b-8192", "Llama3-70b-8192"])
    st.markdown("---")
    st.markdown("**Powered by [LangChain](https://github.com/langchain-ai/streamlit-agent)**")

# User Input
generic_url = st.text_input("Enter a URL (YouTube or Website)", placeholder="https://example.com", label_visibility="visible")

# Initialize LLM
llm = ChatGroq(model=model_name, groq_api_key=api_key)

# Summarization Prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization Button
if st.button("Summarize the Content"):
    if not api_key.strip() or not generic_url.strip():
        st.error("❌ Please provide the required information to proceed.")
    elif not validators.url(generic_url):
        st.error("⚠️ Invalid URL! Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("🔍 Extracting and summarizing content..."):
                # Load data from the source
                docs = []
                if "youtube.com" in generic_url:
                    try:
                        video_id = generic_url.split("v=")[-1]
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        text_content = " ".join([t["text"] for t in transcript])
                        docs = [Document(page_content=text_content)]
                    except Exception as yt_error:
                        st.error(f"🚨 Error processing YouTube video: {yt_error}. Try another URL or check the link.")
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                    )
                    docs = loader.load()

                if docs:
                    # Load summarization chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    
                    st.success("✅ Summary Generated Successfully!")
                    st.write(output_summary)
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
