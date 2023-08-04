import streamlit as st
import os
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from question_answering import ask_and_get_answer, ask_and_get_answer_memory

load_dotenv(find_dotenv(), override=True)

st.write(
    "Variables loaded: \n",
    os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"] and
    os.environ["PINECONE_API_KEY"] == st.secrets["PINECONE_API_KEY"] and
    os.environ["PINECONE_ENV"] == st.secrets["PINECONE_ENV"]
)

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'),
              environment=os.environ.get('PINECONE_ENV'))

nombre_index = 'heru-gpt'
embeddings = OpenAIEmbeddings()


# ---- Loading documents ----
def load_documents(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file} as PDF')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file} as DOCX')
        loader = Docx2txtLoader(file)
    if extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'Loading {file} as TXT')
        loader = TextLoader(file, 'utf-8')
    else:
        print('Document type not supported')
        return None

    data = loader.load()
    return data


# ---- Chunking documents ----
def chunking(documents, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    return chunks


# ---- Embedding documents ----
def create_embeddings(chunks):
    index = pinecone.Index(nombre_index)
    # Borrando vectores anteriores
    index.delete(delete_all=True)
    print(index.describe_index_stats())
    # Guardando nuevos vectores
    vector_store = Pinecone.from_documents(
        chunks, embeddings, index_name=nombre_index)
    return vector_store


# ---- Embedding costs ----
def calculate_embedding_costs(texts):
    import tiktoken
    encoder = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(encoder.encode(page.page_content))
                       for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004


# ============== APPLICATION ==============
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.subheader('Heru GPT 🦄')
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Choose a file", type=['txt'])
        chunk_size = st.number_input(
            'Chunk size', min_value=100, max_value=2048, value=512)
        k = st.number_input('k', min_value=1, max_value=20, value=3)
        add_data = st.button('Ingresar data')

        if uploaded_file and add_data:
            with st.spinner('Leyendo, chunkeando y embebiendo el arhivo ...'):
                bytes_data = uploaded_file.read()
                name_of_file = os.path.join(uploaded_file.name)
                with open(name_of_file, 'wb') as f:
                    f.write(bytes_data)

                datos = load_documents(name_of_file)
                chunks = chunking(datos, chunk_size)
                st.write(
                    f'Tamaño de los chunks: {chunk_size}', f'Número de chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_costs(chunks)
                st.write(f'Costos de embedding: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success(
                    'Archivo cargado, chunkeado y embebido correctamente ✅')

    question = st.text_input('Escribe una pregunta relacionada a Heru')
    if question:
        if 'history' not in st.session_state:
            st.session_state.history = ''
            
        vector_store = Pinecone.from_existing_index(
            index_name=nombre_index, embedding=embeddings)
        answer = ask_and_get_answer_memory(vector_store, question, k, history=st.session_state.history)
        st.text_area('Respuesta', answer)

        results = vector_store.similarity_search(question, k=k)
        result_list = []
        for result in results:
            result_list.append(result.page_content)
            result_string = '\n - \n'.join(result_list)
        st.divider()
        with st.expander("Ver chunks de referencia"):
            st.text(result_string)
        
        value = f'\nHUMAN: {question} \nSYSTEM: {answer}'
        st.session_state.history = f'{value}  \n{st.session_state.history}'
        h = st.session_state.history
        
        with st.expander("Ver historial"):
            st.text(h)