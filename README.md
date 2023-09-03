# Ask a Bot

A Q&A chatbot for you to talk with your PDFs. The app is based on Streamlit, LangChain, and Llama LLM.

To use the app, you will need to install the following dependencies:

`pip install requirements.txt`

You also need to download the respective Llama LLM from [here](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) and put it in the same folder as `app.py`.

Once you have installed the dependencies and downloaded the LLM model, you can run the following command to start the app:

`streamlit run app.py`

You can then interact with the app by uploading a PDF and asking questions about it.
The app will then return the respective answer and render the PDF page where the answer is.

Please note that the performance of this bot is very limited when compared to a ChatGPT-based one due to the size/performance of the underlying LLM.

You can also play with the app online at the Hugging Face Space [ask-a-bot](https://github.com/cmigpereira/ask-a-bot).
