import streamlit as st
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from utils import *


DEFAULT_TEMPLATE = """Assume you are a virtual assistant having a conversation with a human.
{history}
Human: {question}
Assistant:"""

try: 
    with st.container():
        st.title("Smart ChattyBot")
        st.text(
            """
            Welcome to Smart ChattyBot! This version only chats with you without any context. 
            """)

        if "chatbot_chain" not in st.session_state:
            st.session_state["chatbot_chain"] = generate_cb_chain(st.session_state["LLM_MODEL"], DEFAULT_TEMPLATE)

        with st.form("Submit your query here."):
            query = st.text_area("Ask away!")
            submitted = st.form_submit_button("Submit")
            if submitted:
                answer = st.session_state["chatbot_chain"].predict(question=query)
                st.write(answer)
except KeyError:
    st.error("Please choose a LLM provider on the Main page!")