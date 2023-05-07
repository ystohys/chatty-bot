import streamlit as st
from langchain.chains.question_answering import load_qa_chain

from utils import *

with st.container():
        st.title("Context ChattyBot")
        st.text(
            """
            Welcome to Context ChattyBot!
            To get it to answer questions specific to your domain, all you need to 
            do is upload a PDF document containing the relevant information. 
            """)
        domain_doc = st.file_uploader(label="Upload your document in PDF format here")

try:
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = load_qa_chain(st.session_state["LLM_MODEL"], chain_type="stuff")

    if domain_doc:
        with st.spinner("Reading your PDF document..."):
            pdf_bytes = domain_doc.getvalue()
            vector_db = create_vdb_from_pdf(pdf_bytes, st.session_state["EMBEDDING_MODEL"])
            vdb_retriever = vector_db.as_retriever()

    with st.form("Submit your query here."):
        query = st.text_area("Ask away!")
        submitted = st.form_submit_button("Submit")
        if domain_doc and submitted:
            rel_docs = vdb_retriever.get_relevant_documents(query)
            answer = st.session_state["qa_chain"].run(input_documents=rel_docs, question=query)
            st.write(answer)
except KeyError:
    st.error("Please choose a LLM provider on the Main page!")