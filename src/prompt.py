from langchain_core.prompts import ChatPromptTemplate


system_prompt = (
    "You are a helpful medical assistant for answering questions about medical answers. "
    "Use the following retrieved documents to answer the question. "
    "If you don't know the answer, say you don't know. "
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
