import streamlit as st
from utils import *

st.set_page_config(
    page_title="Welcome"
)

st.write("# ChattyBot")

def delete_sessionstate():
    """To wipe existing chains in session_state when we change models."""
    if "chatbot_chain" in st.session_state:
        del st.session_state["chatbot_chain"]
    if "qa_chain" in st.session_state:
        del st.session_state["qa_chain"]
    if "EMBEDDING_MODEL" in st.session_state:
        del st.session_state["EMBEDDING_MODEL"]
    if "LLM_MODEL" in st.session_state:
        del st.session_state["LLM_MODEL"]

model_provider = st.selectbox(label="Choose your LLM/Embedding models:", 
                              options=("--", 
                                       "OpenAI", 
                                       "HuggingFaceHub"
                                       #"Local models (GPT2/S-BERT)"
                                       ),
                            on_change=delete_sessionstate
                            )

st.markdown(
    """
    This ChatBot is primarily powered by OpenAI's GPT3/3.5 models. This means you need to have a
    OpenAI API subscription and an OpenAI API key to use it. However, if you don't like paying for 
    OpenAI, you can also choose the HuggingFaceHub option below, although there is a high chance
    you will easily hit the rate limit with just a few questions. 

    **I am trying to add support for local models, but the models I can save on my laptop performs a lot worse than the OpenAI models.**

    Some things to note:
    - This chatbot works very well with the powerful OpenAI models, the only downside being the subscription costs.
    - The "No-Context" chatbot remembers conversation history, while the "Context" chatbot does not.

    *Prepared by You Sheng*
    """
    )

st.sidebar.success("Choose a chatbot.")

if model_provider == "OpenAI":
    st.session_state["EMBEDDING_MODEL"] = initialize_openai_embedder()
    st.session_state["LLM_MODEL"] = initialize_openai_llm()
    st.success("Models ready.")
elif model_provider == "HuggingFaceHub":
    st.session_state["EMBEDDING_MODEL"] = initialize_hf_embedder(repo="sentence-transformers/all-mpnet-base-v2")
    st.session_state["LLM_MODEL"] = initialize_hf_llm(repo="google/flan-t5-large")
    st.success("Models ready.")
# else:
#     EMBEDDING_MODEL = initialize_local_embedder()
#     LLM_MODEL = initialize_local_llm()
#     st.write("Models downloaded and ready!")


    