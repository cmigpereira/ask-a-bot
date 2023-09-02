import tempfile
import fitz
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PIL import Image
from streamlit_chat import message

st.set_page_config(
    page_title="AskBot",
    page_icon=":robot_face:",
    layout="wide"
)

st.sidebar.title("""
                 Ask A Bot :robot_face: \n Talk with your PDFs
                 """)

st.sidebar.write("""
                ###### A Q&A chatbot for you to talk with your PDFs.
                ###### Upload the PDF you want to talk to and start asking questions. The display will show the page where the answer was found.
                ###### When you upload a new PDF, the chat history is reset for you to start fresh.
                ###### The chatbot is based on Langchain and the Llama language model, which is a large language model trained on the Common Crawl dataset. Obtained from [here](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML).
                ###### The performance of this bot is limited due to its size. For better performance, a larger LLM should be used.
                ###### :warning: Sometimes the Streamlit app will not re-run and refresh the PDF. If this happens, refresh the page.
                ###### Developed by [Carlos Pereira](https://linkedin.com/in/carlos-miguel-pereira/).
                """)

if 'pdf_page' not in st.session_state:
    st.session_state['pdf_page'] = 0

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def update_state():
    """
    Reset state when a new PDF is uploaded
    """
    st.session_state.pdf_page = 0
    st.session_state.chat_history = []
    st.session_state['generated'] = []
    st.session_state['past'] = []

@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Load Llama LLM
    """
    llm_model = CTransformers(
        model="llama-2-13b-chat.ggmlv3.q3_K_L.bin",
        model_type="llama",
        max_new_tokens=150,
        temperature=0.2
    )
    return llm_model

@st.cache_resource(show_spinner=False)
def gen_embeddings():
    """
    Generate embeddings
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                       model_kwargs={'device': 'cpu'})

    return embeddings

def load_pdf(file):
    """
    Load PDF and process for Search
    """
    # create tempfile to load pdf to PyPDFLoader
    temp_file = tempfile.NamedTemporaryFile()
    temp_file.write(file.getbuffer())
    loader = PyPDFLoader(temp_file.name)
    documents = loader.load()
    pdf_file = fitz.open(temp_file.name)
    temp_file.close()

    # split doc into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    #get embedding model
    embeddings = gen_embeddings()
    pdf_search = Chroma.from_documents(texts, embeddings)

    return pdf_search, pdf_file

def generate_chain(pdf_vector, llm):
    """
    Generate Retrieval chain
    """
    chain = ConversationalRetrievalChain.from_llm(llm,
                                chain_type="stuff",
                                retriever=pdf_vector.as_retriever(search_kwargs={"k": 1}),
                                return_source_documents=True)

    return chain

def get_answer(chain, query, chat_history):
    """
    Get an answer from the chain
    """
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    answer = result["answer"]
    # if you want history uncomment the line below
    # st.session_state.chat_history += [(query, answer)]
    st.session_state.pdf_page = list(result['source_documents'][0])[1][1]['page']

    return answer

def render_page_file(file, page):
    """
    Render page from PDF file
    """
    try:
        page = file[page]
    except: # todo: fix this exception handling
        page = file[0]
        st.session_state.pdf_page = 0

    # Render the PDF page as an image
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

    return image

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"],
                                 accept_multiple_files=False,
                                 on_change=update_state)

def app():
    """
    Main app
    """
    if uploaded_file:
        # Load LLM
        with st.spinner('Loading LLM...'):
            llm = load_llm()
        # Load and process the uploaded PDF file
        with st.spinner('Loading PDF...'):
            pdf_vector, pdf_file = load_pdf(uploaded_file)
        with st.spinner('Generating chain...'):
            chain = generate_chain(pdf_vector, llm)

        col1, col2 = st.columns(2)
        with col1:
            # Question and answering
            with st.form(key='question_form', clear_on_submit=True):
                question = st.text_input('Enter your question:', value="", key='text_value')
                submit_question = st.form_submit_button(label="Enter")

            if submit_question:
                with st.spinner('Getting answer...'):
                    answer = get_answer(chain, question,
                                    st.session_state.chat_history)
                st.session_state.past.append(question)
                st.session_state.generated.append(answer)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], is_user=False,
                            avatar_style="bottts", key=str(i))
                    message(st.session_state['past'][i], is_user=True,
                            avatar_style="adventurer", key=str(i) + '_user')

        with col2:
            # Render PDF page
            if pdf_file:
                st.image(render_page_file(pdf_file, st.session_state.pdf_page))

if __name__ == "__main__":
    app()
