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


st.set_page_config(
    page_title="RAG 智能问答助手",
    page_icon="demo/ob-icon.png",
)
st.title("💬 智能问答助手")
st.caption("🚀 使用 OceanBase 向量检索特性和大语言模型能力构建的智能问答机器人")
st.logo("demo/logo.png")

env_table_name = os.getenv("TABLE_NAME", "corpus")
env_llm_base_url = os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

with st.sidebar:
    st.subheader("🔧 设置")
    st.text_input(
        "表名",
        value=env_table_name,
        disabled=True,
        help="用于存放文档及其向量数据的表名，用环境变量 TABLE_NAME 进行设置",
    )
    if env_llm_base_url == "https://open.bigmodel.cn/api/paas/v4/":
        llm_model = st.selectbox(
            "选用的大语言模型",
            ["glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-long"],
            index=0,
        )
    history_len = st.slider(
        "聊天历史长度",
        min_value=0,
        max_value=25,
        value=3,
        help="聊天历史长度，用于上下文理解",
    )
    search_docs = st.checkbox(
        "进行文档检索",
        True,
        help="检索文档以获取更多信息，否则只使用大语言模型回答问题",
    )
    oceanbase_only = st.checkbox(
        "仅限 OceanBase 相关问题",
        True,
        help="勾选后机器人只会回答 OceanBase 有关的问题",
    )
    rerank = st.checkbox(
        "进行文档重排序",
        False,
        help="使用 BGE-M3 对检索的文档进行重排序以提高检索结果的质量，这是一个很慢的过程，请仅在有需要时使用",
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

    history = st.session_state["messages"][-history_len:] if history_len > 0 else []

    it = doc_rag_stream(
        query=prompt,
        chat_history=remove_refs(history),
        universal_rag=not oceanbase_only,
        rerank=rerank,
        llm_model=llm_model,
        search_docs=search_docs,
    )

    with st.status("处理中...", expanded=True) as status:
        for msg in it:
            if not isinstance(msg, str):
                status.update(label="思考完毕！")
                break
            st.write(msg)

    res = StreamResponse(it)

    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("assistant", avatar=avatar_m["assistant"]).write_stream(
        res.generate()
    )

    st.session_state.messages.append({"role": "assistant", "content": res.get_whole()})
