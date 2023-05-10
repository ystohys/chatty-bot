import streamlit as st
import fitz

from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import HuggingFaceHub, OpenAI, HuggingFacePipeline
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Functions to initialize the LLMs and embedding models

@st.cache_data(show_spinner=False)
def initialize_openai_embedder(model="text-embedding-ada-002"):
    return OpenAIEmbeddings(model=model)

@st.cache_data(show_spinner=False)
def initialize_openai_llm(model="text-davinci-003", temp=0.1):
    return OpenAI(model_name=model, temperature=temp)

@st.cache_data(show_spinner=False)
def initialize_hf_embedder(repo, task="feature-extraction"):
    return HuggingFaceHubEmbeddings(repo_id=repo, task=task)

@st.cache_data(show_spinner=False)
def initialize_hf_llm(repo, **kwargs):
    return HuggingFaceHub(repo_id=repo, model_kwargs=kwargs)

@st.cache_data(show_spinner=False)
def initialize_local_embedder(model="all-MiniLM-L6-v2"):
    return SentenceTransformerEmbeddings(model_name=model)

@st.cache_data(show_spinner=False)
def initialize_local_llm(model="facebook/bart-large-mnli", **kwargs):
    return HuggingFacePipeline.from_model_id(model_id=model, task="text2text-generation")


# No_Context_Chat functions

def generate_cb_chain(llm_model, template_str):
    return LLMChain(
        llm = llm_model,
        prompt = PromptTemplate(input_variables=["history", "question"], template=template_str),
        memory = ConversationBufferWindowMemory(k=5, memory_key="history", return_messages=True)
    )

# Context_Chat functions

def parse_pdf_to_docs(file_bytes):
    "Function to read PDF file uploaded to streamlit app and parse the text inside to LangChain Documents"
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    docs_list = []
    for page in pdf.pages():
        text = page.get_text()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        for chunk in text_splitter.split_text(text):
            curr_doc = Document(page_content=chunk, 
                                metadata={"page_number": page.number+1, "total_pages": len(pdf)})
            docs_list.append(curr_doc)
    return docs_list

@st.cache_resource(show_spinner=False)
def create_vdb_from_pdf(pdf_bytes, _embedding_model, persist=False):
    docs_list = parse_pdf_to_docs(pdf_bytes)
    vdb = FAISS.from_documents(docs_list, _embedding_model)
    return vdb