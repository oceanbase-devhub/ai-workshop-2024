import os
import dotenv

dotenv.load_dotenv()

from typing import Iterator, Union
from rag.doc_rag import doc_rag_stream

import streamlit as st
from langchain_core.messages import BaseMessageChunk


class StreamResponse:
    """
    StreamResponse is a class that helps to stream the response from the chatbot.
    """

    def __init__(self, chunks: Iterator[BaseMessageChunk] = []):
        self.chunks = chunks
        self.__whole_msg = ""

    def generate(
        self,
        *,
        prefix: Union[str, None] = None,
        suffix: Union[str, None] = None,
    ) -> Iterator[str]:
        if prefix:
            yield prefix
        for chunk in self.chunks:
            self.__whole_msg += chunk.content
            yield chunk.content
        if suffix:
            yield suffix

    def get_whole(self) -> str:
        return self.__whole_msg


st.set_page_config(page_title="RAG ChatBot", page_icon="demo/ob-icon.png")
st.title("💬 RAG ChatBot")
st.caption("🚀 A chatbot powered by OceanBase, ZhipuAI and Streamlit")
st.logo("demo/logo.png")

env_table_name = os.getenv("TABLE_NAME", "corpus")
env_llm_base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

with st.sidebar:
    st.subheader("🔧Settings")
    st.text_input(
        "TABLE_NAME",
        value=env_table_name,
        disabled=True,
        help="The table name of the data in the database. Which should be set by TABLE_NAME env variable.",
    )
    if env_llm_base_url == "https://open.bigmodel.cn/api/paas/v4/":
        llm_model = st.selectbox(
            "LLM Model",
            ["glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-long"],
            index=0,
        )
    history_len = st.slider(
        "Chat History Length",
        min_value=0,
        max_value=25,
        value=3,
        help="The length of the chat history.",
    )
    search_docs = st.checkbox(
        "Search Docs",
        True,
        help="Search the documents to answer the questions.",
    )
    oceanbase_only = st.checkbox(
        "Only OceanBase",
        True,
        help="Only answer OceanBase related questions",
    )
    rerank = st.checkbox(
        "Rerank Docs",
        False,
        help="Rerank retrieved documents using the bge-m3 model to enhance generation, which is quite a slow process.",
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "您好，请问有什么可以帮助您的吗？"}
    ]

avatar_m = {
    "assistant": "demo/ob-icon.png",
    "user": "🧑‍💻",
}

for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=avatar_m[msg["role"]]).write(msg["content"])


def remove_refs(history: list[dict]) -> list[dict]:
    """
    Remove the references from the chat history.
    This prevents the model from generating its own reference list.
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"].split("根据向量相似性匹配检索")[0],
        }
        for msg in history
    ]


if prompt := st.chat_input("请输入您想咨询的问题..."):
    st.chat_message("user", avatar=avatar_m["user"]).write(prompt)

    history = st.session_state["messages"][-history_len:]

    it = doc_rag_stream(
        query=prompt,
        chat_history=remove_refs(history),
        universal_rag=not oceanbase_only,
        rerank=rerank,
        llm_model=llm_model,
        search_docs=search_docs,
    )

    with st.status("Processing", expanded=True) as status:
        for msg in it:
            if not isinstance(msg, str):
                status.update(label="Finish thinking!")
                break
            st.write(msg)

    res = StreamResponse(it)

    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("assistant", avatar=avatar_m["assistant"]).write_stream(
        res.generate()
    )

    st.session_state.messages.append({"role": "assistant", "content": res.get_whole()})
