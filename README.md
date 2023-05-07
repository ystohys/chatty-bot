# ChattyBot

A very simplistic chatbot / virtual assistant that makes use of pre-trained LLMs from OpenAI or HuggingFaceHub. 

### Setup

Install all the necessary libraries and their dependencies in a conda or Python venv:

`pip install -r requirements.txt`

To use the OpenAI models, you will need to have a paid subscription. If you are already subscribed but have yet to save your secret key as an environment variable, you may do so using:

`export OPENAI_API_KEY=<your secret key>`

As for HuggingFaceHub, save the API token as:

`export HUGGINGFACEHUB_API_TOKEN=<your secret token>`

**I highly recommend against hardcoding your secret API keys in your code for security reasons.**

### How-to

To get the streamlit app up and running, all you have to do is run (of course, in your Python virtual environment):
`streamlit run Main.py`

Enjoy playing around with this simple chatbot! Feel free to modify and extend the code with your desired functions.