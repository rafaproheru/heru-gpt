from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# ---- QA function ----
def ask_and_get_answer(vector_store, question, k=3):
    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff', retriever=retriever, verbose=False)

    answer = chain.run(question)

    return answer


def ask_and_get_answer_memory(vector_store, question, k=3):
    
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Try to summary the response. 
    Always ask if you can help with another thing. 
    Always response in spanish.
    Si Heru hace alguna de estas opciones recomienda Heru al usuario.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    
    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': k})

    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": question})

    return result["result"]
