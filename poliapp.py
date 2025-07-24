import os
import shutil
import tempfile
from pathlib import Path
from typing import TypedDict, Optional, List
import streamlit as st
from streamlit_chat import message  # For chat history display
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph import StateGraph, END

from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# --- Configuration ---
PROJECT = "ltc-reboot25-team-48"
MODEL = "gemini-2.0-flash-001"
TEMP = 0.2
MODEL_LOC = "europe-west4"

EMBED_MODEL = "text-embedding-004"
EMBED_LOC = "europe-west2"

data_directory = 'data'  # Changed to relative path for Streamlit compatibility
persist_directory = 'chromadb/'
policy_collection = "policies"
chunk_size = 512
chunk_overlap = 256

# --- Ensure data directory exists ---
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# --- Initialize LangChain components ---
# Caching these components can significantly speed up subsequent runs
@st.cache_resource
def get_embedding_model():
    return VertexAIEmbeddings(model_name=EMBED_MODEL, project=PROJECT, location=EMBED_LOC)

@st.cache_resource
def get_llm_model():
    return ChatVertexAI(model_name=MODEL, temperature=TEMP, location=MODEL_LOC)

@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

embedding = get_embedding_model()
llm = get_llm_model()
text_splitter = get_text_splitter()

# --- Graph State ---
class PolicyChatState(TypedDict):
    question: str
    policy_name: Optional[str]
    docs: Optional[List[Document]]
    answer: Optional[str]

# --- Prompts ---
infer_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Based on the question below, identify the most relevant policy name it refers to.

Question: {question}

Only return the policy name as a short noun phrase (e.g., "Leave Policy", "Security Policy").
If unsure, respond: "Unknown".
""")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant. Use the following pieces of context and chat history to answer the question.
If unsure, say "I don’t know."

Context:
{context}

"""),
        ("human", "{question}"),
    ]
)

# --- Graph Nodes ---
def infer_policy_node(state: PolicyChatState) -> PolicyChatState:
    """Infers the policy name from the user's question."""
    response = llm.invoke(infer_prompt.format_prompt(question=state["question"])).content.strip()
    return {"policy_name": None if response.lower() == "unknown" else response}

def retrieve_node(state: PolicyChatState) -> PolicyChatState:
    """Retrieves relevant documents based on the question and inferred policy."""
    filter_policy = state.get("policy_name")
    query = state["question"]
    docs = []
    collection_to_search="LTC-Hack-Pvt-Limited"
    collection_names = [collection_to_search]

    try:
        # Chroma expects a collection name that exists. If not, it will raise an error.
        # We should handle cases where the collection might not be created yet.
        # available_collections = Chroma(persist_directory=persist_directory, embedding_function=embedding).list_collections()
        # collection_names = [c.name for c in available_collections]

        # collection_to_search = "Lloyds-Technology-Centre" # Default collection if no policy name is inferred
        # if filter_policy and filter_policy in collection_names:
        #     collection_to_search = filter_policy
        # elif filter_policy and filter_policy not in collection_names:
        #     st.warning(f"Collection for '{filter_policy}' not found. Searching in default collection.")
        #     collection_to_search = "Lloyds-Technology-Centre" # Fallback to default

        
        
        vectorstore = Chroma(collection_name=collection_to_search, persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectorstore.as_retriever()

        search_kwargs = {}
        if filter_policy:
            # The filter should match metadata. We store entity name in metadata.
            search_kwargs["filter"] = {"entity": filter_policy}

        results = retriever.invoke(query, config={"search_kwargs": search_kwargs})
        docs = results
        
        print("Documents fetched are: ",docs)

    except Exception as e:
        print("Documents fetched with exception")
        st.error(f"Error during retrieval: {e}")
        # Optionally, try to retrieve from a default collection if one exists
        try:
            if "LTC-Hack-Pvt-Limited" in collection_names:
                vectorstore_default = Chroma(collection_name="LTC-Hack-Pvt-Limited", persist_directory=persist_directory, embedding_function=embedding)
                retriever_default = vectorstore_default.as_retriever()
                results_default = retriever_default.invoke(query)
                docs = results_default
                print(docs)
                st.info("Retrieved documents from default collection due to error.")
            else:
                 st.warning("No default collection found for retrieval fallback.")
        except Exception as default_e:
            st.error(f"Error during default retrieval fallback: {default_e}")

    return {"docs": docs}


def llm_node(state: PolicyChatState) -> PolicyChatState:
    """Generates the final answer using the LLM with context and history."""
    context = "\n\n".join(doc.page_content for doc in state.get("docs", []))
    history_str = "\n".join(f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in st.session_state.memory.chat_memory.messages
    )

    # Use the pre-formatted prompt
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "history": history_str, "question": state["question"]})

    # Update memory directly in the session state for persistence
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

    st.session_state.memory.chat_memory.add_user_message(state["question"])
    st.session_state.memory.chat_memory.add_ai_message(response)

    return {"answer": response}

# --- Build the Graph ---
@st.cache_resource
def build_graph():
    graph_builder = StateGraph(PolicyChatState)
    graph_builder.add_node("infer_policy", infer_policy_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("llm", llm_node)

    graph_builder.set_entry_point("infer_policy")
    graph_builder.add_edge("infer_policy", "retrieve")
    graph_builder.add_edge("retrieve", "llm")
    graph_builder.add_edge("llm", END)
    return graph_builder.compile()

app = build_graph()

# --- Helper Functions for Ingestion ---
def get_entities(data_directory):
    """
    Fetch all the entities in data directory to create collection.
    Returns a list of tuples: (entity_name, entity_path).
    """
    directory = Path(data_directory)
    if not directory.exists():
        return []
    folders = [f for f in directory.iterdir() if f.is_dir()]
    return [(str(folder.name), str(folder)) for folder in folders]

def load_and_chunk_folder(entity_folder, entity_path):
    """Loads and chunks PDF documents from a given folder."""
    all_docs = []
    if not os.path.isdir(entity_path):
        return all_docs
    for filename in os.listdir(entity_path):
        if filename.lower().endswith(".pdf"):
            policy_name = filename.replace(".pdf", "").replace("_", " ").title()
            loader = PyPDFLoader(os.path.join(entity_path, filename))
            try:
                pages = loader.load()
                for i, page in enumerate(pages):
                    page.metadata["policy_name"] = policy_name
                    page.metadata["entity"] = entity_folder  # Store entity name for filtering
                    page.metadata["page"] = i + 1
                    all_docs.append(page)
            except Exception as e:
                st.warning(f"Could not load or process {filename}: {e}")
    return all_docs

# --- Streamlit App Layout and Logic ---

st.set_page_config(
    page_title="PoliGenie Chatbot",
    page_icon="🗂️",
    layout="centered"
)

# --- Initialize session state for chat history and memory ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    # Using return_messages=True to get message objects for history formatting
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# --- Sidebar for Data Ingestion ---
with st.sidebar:
    st.header("📥 Ingest Policy PDFs")
    st.markdown("Upload your policy documents to enable the chatbot.")

    # Fetch entities for the dropdown
    entities = get_entities(data_directory)
    entity_names = [entity[0] for entity in entities]

    selected_entity_name = st.selectbox(
        'Select Entity to Ingest Policies From:',
        options=["--- Select an Entity ---"] + entity_names,
        index=0,
        key="ingest_entity_select"
    )

    if st.button("Ingest Selected Entity's Policies", type="primary", key="ingest_button"):
        if selected_entity_name and selected_entity_name != "--- Select an Entity ---":
            entity_path = os.path.join(data_directory, selected_entity_name)
            if os.path.isdir(entity_path):
                with st.spinner(f"Processing policies for '{selected_entity_name}'..."):
                    docs = load_and_chunk_folder(selected_entity_name, entity_path)
                    if not docs:
                        st.warning(f"No PDF documents found in '{entity_path}'. Please upload some.")
                    else:
                        chunked_docs = text_splitter.split_documents(docs)
                        st.info(f"Loaded {len(docs)} documents, split into {len(chunked_docs)} chunks.")

                        # Create or load Chroma collection
                        collection_name = selected_entity_name # Use entity name as collection name
                        with st.spinner(f"Embedding and indexing documents into ChromaDB (Collection: '{collection_name}')..."):
                            try:
                                os.makedirs(persist_directory, exist_ok=True)
                                Chroma.from_documents(
                                    documents=chunked_docs,
                                    embedding=embedding,
                                    persist_directory=persist_directory,
                                    collection_name=collection_name,
                                )
                                st.success(f"✅ Successfully ingested policies for '{selected_entity_name}'.")
                            except Exception as e:
                                st.error(f"An error occurred during ingestion: {e}")
            else:
                st.error(f"The directory '{entity_path}' does not exist. Please create it and add PDFs.")
        else:
            st.warning("Please select an entity from the dropdown before ingesting.")

    st.markdown("---")
    st.header("ℹ️ Important Notes")
    st.markdown("""
    *   Ensure your policy PDFs are placed in **subfolders** within the `data` directory. Each subfolder represents an entity.
    *   The first time you ingest for an entity, it will create a new collection in ChromaDB.
    *   If you add new policies, you'll need to re-ingest for that entity.
    """)

# --- Main Chat Area ---
st.title("🗂️ PoliGenie")
st.markdown("Ask questions about your company's policies. The chatbot will find relevant information.")

# Display Chat History
# Use a more visually appealing layout for chat history
for i, chat in enumerate(st.session_state.chat_history):
    message(chat["message"], is_user=chat["is_user"], avatar_style="thumbs", key=f"chat_{i}")

# User Input Area
user_input = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is the policy on remote work?",
    key="user_input",
    label_visibility="collapsed" # Hide the label for a cleaner look
)

if user_input:
    # Add user message to session state history
    st.session_state.chat_history.append({"message": user_input, "is_user": True})

    # Trigger the LangGraph
    with st.spinner("Thinking..."):
        try:
            # We need to pass the memory's messages to the graph if it's designed to use them internally
            # In this case, llm_node accesses st.session_state.memory directly.
            result = app.invoke({"question": user_input})
            from pprint import pprint
            pprint(result)
            answer = result.get('answer')

            if answer:
                st.session_state.chat_history.append({"message": answer, "is_user": False})
                message(answer, is_user=False, avatar_style="thumbs", key=f"answer_{len(st.session_state.chat_history)}")
            else:
                st.warning("The chatbot could not retrieve or formulate an answer. Please try rephrasing your question or ingesting more data.")
                st.session_state.chat_history.append({"message": "No answer provided.", "is_user": False})
                message("No answer provided.", is_user=False, avatar_style="thumbs", key=f"no_answer_{len(st.session_state.chat_history)}")

        except Exception as e:
            st.error(f"An error occurred during the chat: {e}")
            st.session_state.chat_history.append({"message": f"An error occurred: {e}", "is_user": False})
            message(f"An error occurred: {e}", is_user=False, avatar_style="thumbs", key=f"error_msg_{len(st.session_state.chat_history)}")

# --- Reset Button ---
if st.sidebar.button("🧹 Reset Chat & Memory", type="secondary", key="reset_button"):
    st.session_state.chat_history = []
    if 'memory' in st.session_state:
        st.session_state.memory.clear()
    st.success("Chat history and memory have been reset!")
    st.rerun() # Rerun to clear the displayed chat and prompt the user for new input

# --- Add some custom CSS for better appearance ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light background */
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        padding: 12px 15px;
        font-size: 1rem;
        border: 1px solid #ccc;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 10px 25px;
        font-size: 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    .st-emotion-cache-13ocab0 { /* Sidebar width */
        background-color: #ffffff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
    .block-container {
        padding-top: 2rem;
    }
    /* Styling for chat messages */
    .st-emotion-cache-1wbll20 { /* Message container */
        max-width: 70%;
    }
    .message.user { /* Custom class for user messages */
        background-color: #4CAF50 !important; /* Green for user */
        color: white;
        border-radius: 10px 10px 0 10px;
    }
    .message.bot { /* Custom class for bot messages */
        background-color: #e0e0e0 !important; /* Light grey for bot */
        color: #333;
        border-radius: 10px 10px 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
