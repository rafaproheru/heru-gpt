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


def ask_and_get_answer_memory(vector_store, question, k=3, history=""):
    
    template = """
    You are a support agent in an accounting company called Heru, you have to answer kindly and always offer more help.
    
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    try to summarize the answer in 4 sentences or less.
    Always ask if you can help with another thing. 
    ALWAYS REPLY IN SPANISH.
    Always provide a professional and warm response.
    También usa la conversación anterior como contexto para contestar HUMAN es el usuario y tu eres SYSTEM.
    
    {context}
    
    Question: {question}
    
    Conversacion anterior: {history}
    
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        partial_variables={"history": history},
        template=template,
    )
    
    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': k})

    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": question})

    return result["result"]
