import streamlit as st
from openai import OpenAI
from pymongo.mongo_client import MongoClient
import time

# import os
# from dotenv import load_dotenv

# load_dotenv()


# Define GLOBAL VARS
# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PRICE_PER_TOKEN = 0.02 / 1_000_000

# MongoDB Vector Search
VECTOR_NUM_CANDIDATES = 150
VECTOR_RESULT_LIMIT = 10
VECTOR_PENALTY = 1
FULL_TEXT_PENALTY = 1

# Chat completion model
CHAT_MODEL = "gpt-4o-mini"
CHAT_INPUT_PRICE = 0.15 / 1_000_000
CHAT_OUTPUT_PRICE = 0.6 / 1_000_000
CHAT_TEMPERATURE = 0.2
CHAT_MAX_OUTPUT_TOKENS = 500

# DB and API connections
# Connect to OpenAI
openapi_client = OpenAI(api_key=st.secrets.openai.api_key)
# openapi_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))


# Connect to MongoDB and cache the connection
@st.cache_resource
def init_connection():
    return MongoClient(st.secrets.mongodb.uri)
    # return MongoClient(os.getenv("MONGODB_URI"))


client = init_connection()
# Connect to DB and collection
db = client[st.secrets.mongodb.db_name]
# db = client[os.getenv("MONGODB_DB_NAME")]
mongo_collection = db.get_collection(st.secrets.mongodb.collection_name)
# mongo_collection = db.get_collection(os.getenv("MONGODB_COLLECTION_NAME"))


@st.cache_data(ttl="1d", show_spinner=False)
def embed_query(query: str) -> list[float]:
    """Takes a query string and returns the vector embedding via openai API,
    results are cached for 1 day"""
    embedding_result = openapi_client.embeddings.create(
        model=EMBEDDING_MODEL, input=[query]
    )
    query_embedding = embedding_result.data[0].embedding
    return query_embedding


@st.cache_data(ttl="1d", show_spinner=False)
def simple_vector_search(query_embedding: list[float]) -> list[dict]:
    """Takes a query embedding and performs vector search on a MongoDB collection,
    returns relevant docs. Results are cached for 1 day."""
    # Define mongodb query pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": query_embedding,
                "numCandidates": VECTOR_NUM_CANDIDATES,
                "limit": VECTOR_RESULT_LIMIT,
            }
        },
        {
            "$project": {
                "_id": 0,
                "doc_id": 1,
                "chunk_id": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    # run pipeline
    mongo_result = list(mongo_collection.aggregate(pipeline))
    return mongo_result


def hybrid_search(query: str, query_embedding: list[float]) -> list[dict]:
    """Takes a query string and query embedding and performs a hybrid search
    using both vector search and full-text search. Returns a list of relevant docs."""
    mongo_result = list(
        mongo_collection.aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embeddings",
                        "queryVector": query_embedding,
                        "numCandidates": VECTOR_NUM_CANDIDATES,
                        "limit": VECTOR_RESULT_LIMIT,
                    }
                },
                {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                {
                    "$addFields": {
                        "vs_score": {
                            "$divide": [1.0, {"$add": ["$rank", VECTOR_PENALTY, 1]}]
                        }
                    }
                },
                # {"$project": {"vs_score": 1, "_id": "$docs._id", "text": "$docs.text"}},
                {
                    "$project": {
                        "vs_score": 1,
                        "doc_id": "$docs.doc_id",
                        "chunk_id": "$docs.chunk_id",
                        "text": "$docs.text",
                    }
                },
                {
                    "$unionWith": {
                        "coll": st.secrets.mongodb.collection_name,  # os.getenv("MONGODB_COLLECTION_NAME"),
                        "pipeline": [
                            {"$search": {"phrase": {"query": query, "path": "text"}}},
                            {"$limit": 20},
                            {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                            {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                            {
                                "$addFields": {
                                    "fts_score": {
                                        "$divide": [
                                            1.0,
                                            {"$add": ["$rank", FULL_TEXT_PENALTY, 1]},
                                        ]
                                    }
                                }
                            },
                            {
                                "$project": {
                                    "fts_score": 1,
                                    # "_id": "$docs._id",
                                    "doc_id": "$docs.doc_id",
                                    "chunk_id": "$docs.chunk_id",
                                    "text": "$docs.text",
                                }
                            },
                        ],
                    }
                },
                {
                    "$group": {
                        "_id": {"doc_id": "$doc_id", "chunk_id": "$chunk_id"},
                        "text": {"$first": "$text"},
                        "vs_score": {"$max": "$vs_score"},
                        "fts_score": {"$max": "$fts_score"},
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "text": 1,
                        "chunk_id": 1,
                        "doc_id": 1,
                        "vs_score": {"$ifNull": ["$vs_score", 0]},
                        "fts_score": {"$ifNull": ["$fts_score", 0]},
                    }
                },
                {
                    "$project": {
                        "score": {"$add": ["$fts_score", "$vs_score"]},
                        "_id": 1,
                        "text": 1,
                        "chunk_id": 1,
                        "doc_id": 1,
                        "vs_score": 1,
                        "fts_score": 1,
                    }
                },
                {"$sort": {"score": -1}},
                {"$limit": 10},
            ]
        )
    )
    return mongo_result


tools = [
    {
        "type": "function",
        "function": {
            "name": "make_bar_chart",
            "description": """Generates a bar chart from a list of x and y values. 
            The x values are the 'vectorSearchscores' as floats and the y values are the 
            document IDs which should be strings - both from the returned 'docs'.
            Call this function when a user asks for a bar chart of the doc ids and vector search scores.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "vector_search_score": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The vector search scores for each document.",
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The document IDs for each document.",
                    },
                },
                "required": ["vector_search_score", "doc_ids"],
                "additionalProperties": False,
            },
        },
    }
]


@st.cache_data(ttl="1d", show_spinner=False)
def chat_completion(query: str, docs: list[dict]) -> str:
    """Takes a query string, and list of retrieved docs
    and returns a chat completion. Results are cached for 1 day."""
    completion_result = openapi_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        max_tokens=CHAT_MAX_OUTPUT_TOKENS,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful research assistant. 
             Please only base your answers on the list of documents provided.
             Ignore /n characters as they may represent newlines. 
             Do not use external sources. Note the query will follow the word query and the documents will follow the word documents.
             Please cite the 'doc_id' and 'chunk_id' of the document you are referencing for each part of your answer.""",
            },
            {
                "role": "user",
                "content": f"Based on the following documents: {docs}, answer the following query: {query}",
            },
        ],
        tools=tools,
        tool_choice="auto",
    )

    usage = (
        completion_result.usage.prompt_tokens * CHAT_INPUT_PRICE
        + completion_result.usage.completion_tokens * CHAT_OUTPUT_PRICE
    )

    if completion_result.choices[0].finish_reason == "tool_calls":
        st.session_state.tool_call = completion_result.choices[0].message.tool_calls[0]
        return completion_result.choices[0].message.tool_calls[0], usage
    else:
        return completion_result.choices[0].message.content, usage
