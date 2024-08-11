import streamlit as st
import vector_search
import time
import plotly.express as px
import pandas as pd
import openai
import json

# from dotenv import load_dotenv

# load_dotenv()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_call" not in st.session_state:
    st.session_state.tool_call = ""


st.title("Vector::Wise")
st.divider()


# Could make function call in openAI more general to specify just x and y vals to return
def make_bar_chart(x: list[float], y: list[str]) -> None:
    df = pd.DataFrame({"Vector_Search_Score": x, "Doc_ID": y})
    fig = px.bar(
        df,
        x="Vector_Search_Score",
        y="Doc_ID",
        labels={"Vector_Search_Score": "Vector Search Score", "Doc_ID": "Document ID"},
        title="Vector Search Scores by Document ID",
    )
    st.plotly_chart(fig)
    st.dataframe(df)
    st.download_button(
        "Download",
        df.to_csv(index=False).encode("utf-8"),
        "vector_search.csv",
        "text/csv",
    )


if isinstance(
    st.session_state.tool_call,
    openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCall,
):
    # st.write("Yay its a tool call!")
    arguments = json.loads(st.session_state.tool_call.function.arguments)
    make_bar_chart(arguments["vector_search_score"], arguments["doc_ids"])

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

placeholder = st.empty()


# Add an operator :: so that before the operator is the documents or topic to search, and after is the task to complete
# Example: "Home decor :: What are the latest trends in home decor?"


# Investigate tools and function calls
# Could be used with mongodb pre filtering, especially for timestamps and dates...
def query_pipeline() -> None:
    st.session_state.tool_call = ""
    with placeholder.status(label="Searching for answers...", expanded=True) as status:
        status_message = st.empty()

        # query = st.session_state.query
        if "::" in st.session_state.query:
            topic, query = [
                word.strip() for word in st.session_state.query.split("::", 1)
            ]
        else:
            topic = st.session_state.query
            query = st.session_state.query
        st.session_state.messages.append(
            {"role": "user", "content": f"Topic: {topic} - {query}"}
        )

        status_message.write("Emedding query...")
        query_embedding = vector_search.embed_query(topic)

        status_message.write("Searching database for relevant documents...")
        results = vector_search.simple_vector_search(query_embedding)

        status_message.write("Generating chat completion...")
        completion_result, usage = vector_search.chat_completion(query, results)

        status.update(label="Done!", state="complete", expanded=False)
        time.sleep(1)
        st.session_state.messages.append(
            {"role": "ai", "content": f"{completion_result} ... cost: ${usage:.6f}"}
        )


def doc_retrieval_pipeline() -> None:
    with placeholder.status(label="Retrieving documents...", expanded=True) as status:
        status_message = st.empty()

        query = st.session_state.doc_retrieval
        st.session_state.messages.append({"role": "user", "content": query})

        status_message.write("Emedding query...")
        query_embedding = vector_search.embed_query(query)

        status_message.write("Searching database for relevant documents...")
        results = vector_search.simple_vector_search(query_embedding)

        status.update(label="Done!", state="complete", expanded=False)
        time.sleep(1)
        st.session_state.messages.append({"role": "ai", "content": f"{results} "})


def doc_retrieval_hybrid_pipeline() -> None:
    with placeholder.status(label="Retrieving documents...", expanded=True) as status:
        status_message = st.empty()

        query = st.session_state.hybrid
        st.session_state.messages.append({"role": "user", "content": query})

        status_message.write("Emedding query...")
        query_embedding = vector_search.embed_query(query)

        status_message.write("Searching database for relevant documents...")
        results = vector_search.hybrid_search(query, query_embedding)

        status.update(label="Done!", state="complete", expanded=False)
        time.sleep(1)
        st.session_state.messages.append({"role": "ai", "content": f"{results} "})


col1, col2 = st.columns(2)
doc_toggle = col1.toggle("Document retrieval")
hybrid_toggle = col2.toggle("Hybrid search")

if doc_toggle:
    st.chat_input(
        "Retrieve documents", on_submit=doc_retrieval_pipeline, key="doc_retrieval"
    )
elif hybrid_toggle:
    st.chat_input(
        "Retrieve documents - hybrid",
        on_submit=doc_retrieval_hybrid_pipeline,
        key="hybrid",
    )
else:
    st.chat_input("Find answers", on_submit=query_pipeline, key="query")
