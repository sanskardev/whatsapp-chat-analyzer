import streamlit as st
import re
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv




def set_retriever(vectorstore, clean_chat_str):
    # retrieval
    retriever = vectorstore.as_retriever()
    retrieval  = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    )
    full_context_retrieval = RunnableParallel(
        {"context": lambda _: clean_chat_str, "input": RunnablePassthrough()}
    )
    return retrieval, full_context_retrieval

def clean_chats(chat_txt_file):
    raw_chat_str = chat_txt_file.getvalue().decode("utf-8")

    #cleaning
    
    remove_line = "Messages and calls are end-to-end encrypted. Only people in this chat can read, listen to, or share them. Learn more."
    re_remove_pattern = f".*{remove_line}.*\n?"
    clean_chat_str1 = re.sub(re_remove_pattern, "", raw_chat_str)
    
    
    re_group_pattern = re.compile(r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?(?:\s*-\s*|\s+)([^:]+):\s*(.+)$")
    
    
    messages = []
    clean_chat_str = ""
    
    for line in clean_chat_str1.splitlines():
        
        match = re_group_pattern.match(line.strip())
        
        if match:
            date, time, sender, message = match.groups()
            if message not in ('null',
                            'Missed video call',
                            'Missed voice call',
                            'Missed group video call',
                            'Missed group voice call'):
                messages.append({
                    "date": date,
                    "time": time,
                    "sender": sender,
                    "message": message
                })
                formatted_line = f"{date}, {time} - {sender}: {message}\n"
                clean_chat_str += formatted_line

    return messages, clean_chat_str

def create_embeddings(messages):
    #creating docs      
    docs = [
        Document(
            page_content=m["message"],
            metadata={
                "date": m["date"],
                "time": m["time"],
                "sender": m["sender"]
            }
        )
        for m in messages
    ]
    
    #creating embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def create_chains(retrieval, full_context_retrieval):
    # creating qa_chain and summary_chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert on answering specific questions about who said what, dates, details in the chat. Use the following context to answer questions.\n\n{context}"),
            ("human", "{input}"),
        ]
    )
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert on summarizing the entire conversation or giving overviews. Use the following context to answer questions.\n\n{context}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = retrieval | qa_prompt | llm
    summary_chain = full_context_retrieval | summary_prompt | llm

    #router_chain
    route_system = """
    Given a raw text input to a language model, select the model prompt best suited for the input. You will be given the names of the available prompts and a description of what the prompt is best suited for.

    << CANDIDATE PROMPTS >>

    qa_prompt: prompt seeking answer for specific questions about who said what, dates, details in the chat.
    summary_prompt: prompt seeking summary, overview or analyzing the entire conversation.

    << INPUT >>
    {input}

    << OUTPUT >>
    Return only one word and the value being either "qa_prompt" or "summary_prompt" depending on which is the most suitable.
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", route_system),
            ("human", "{input}"),
        ]
    )
    router_chain = route_prompt | llm

    #master_chain
    master_chain = RunnableBranch(
        (lambda x: "qa_prompt" in router_chain.invoke(x).content.lower(), qa_chain),
        (lambda x: "summary_prompt" in router_chain.invoke(x).content.lower(), summary_chain),
        RunnableLambda(lambda x: "Sorry, I don't know how to handle that.")  # default
    )
    
    return master_chain


def get_result(prompt: str, vectorstore, clean_chat_str):
    with st.spinner("Please wait..."):
        retrieval, full_context_retrieval = set_retriever(vectorstore, clean_chat_str)
        master_chain = create_chains(retrieval, full_context_retrieval)
        result = master_chain.invoke(prompt)
        return result

#embedding model config
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


if __name__ == '__main__':
    
    load_dotenv()

    #llm
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")    
    
    with st.sidebar:
        help_url = "https://faq.whatsapp.com/1180414079177245/?helpref=faq_content&cms_platform=android"
        st.write("[Learn how to export your WhatsApp chat.](%s)" % help_url)
        
        st.divider(width = 50)
        
        @st.dialog("Security Information")
        def security_info():
            st.write("Your chat is not stored anywhere (I don't have the money to store your data).")
            st.write("However, this application is using Google's model gemini-2.5-flash. So, if you don't trust Google, don't give it sensitive chats. I, personally, wouldn't.")
        if "security_info" not in st.session_state:
            if st.button("Security Information"):
                security_info()
 
    
    st.title("Ask your WhatsApp anything!")
    

    chat_txt_file = st.file_uploader("Upload your chat txt file.", type="txt", accept_multiple_files=False)

    if chat_txt_file is not None:

        if 'is_analyzed' not in st.session_state:
            
            with st.spinner("Analyzing your chats..."):
                messages, st.session_state.clean_chat_str = clean_chats(chat_txt_file)
                st.session_state.vectorstore = create_embeddings(messages)
                st.session_state.is_analyzed = True
            
        #prompt box
        st.divider(width = 50)
        with st.form(key = "prompt_box"):
            col1, col2 = st.columns([5,1])
            with col1:
                prompt = st.text_input("", label_visibility = "collapsed", placeholder = "Ask anything about this conversation.", icon="❔")
            with col2:
                submit_button = st.form_submit_button(label="Submit")

        
        if prompt:
            result = get_result(prompt, st.session_state.vectorstore, st.session_state.clean_chat_str)
            st.subheader("Answer")
            st.text(result.content)
    
    
    elif 'is_analyzed' in st.session_state:
        del st.session_state.is_analyzed