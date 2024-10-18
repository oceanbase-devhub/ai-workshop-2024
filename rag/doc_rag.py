import re
import os
import time
import json
import random

from typing import Iterator, Optional, Union
from langchain_core.messages import AIMessageChunk
from rag.embeddings import get_embedding
from rag.documents import Document, DocumentMeta, component_mapping as cm
from agents.rag_agent import rag_agent
from agents.intent_guard_agent import guard_agent as iga
from agents.comp_analyzing_agent import component_analyzing_agent as caa
from connection import connection_args
from sqlalchemy import Column, Integer

from langchain_community.vectorstores import OceanBase

embeddings = get_embedding()

vs = OceanBase(
    embedding_function=embeddings,
    table_name=os.getenv("TABLE_NAME", "corpus"),
    connection_args=connection_args,
    metadata_field="metadata",
    extra_columns=[Column("component_code", Integer, primary_key=True)],
    echo=os.getenv("ECHO") == "true",
)

doc_cite_pattern = r"(\[+\@(\d+)\]+)"


def doc_search(
    query: str,
    partition_names: Optional[list[str]] = None,
    limit: int = 10,
) -> list[Document]:
    """
    Search for documents related to the query.
    """
    docs = vs.similarity_search(
        query=query,
        k=limit,
        partition_names=partition_names,
    )

    return docs


def doc_search_by_vector(
    vector: list[float],
    partition_names: Optional[list[str]] = None,
    limit: int = 10,
) -> list[Document]:
    """
    Search for documents related to the query.
    """
    docs = vs.similarity_search_by_vector(
        embedding=vector,
        k=limit,
        partition_names=partition_names,
    )

    return docs


supported_components = cm.keys()

replacers = [
    (r"^.*oceanbase-doc", "https://github.com/oceanbase/oceanbase-doc/blob/V4.3.3"),
    (r"^.*ocp-doc", "https://github.com/oceanbase/ocp-doc/blob/V4.3.0"),
    (r"^.*odc-doc", "https://github.com/oceanbase/odc-doc/blob/V4.3.1"),
    (r"^.*oms-doc", "https://github.com/oceanbase/oms-doc/blob/V4.2.5"),
    (r"^.*obd-doc", "https://github.com/oceanbase/obd-doc/blob/V2.10.0"),
    (
        r"^.*oceanbase-proxy-doc",
        "https://github.com/oceanbase/oceanbase-proxy-doc/blob/V4.3.0",
    ),
    (r"^.*?ob-operator/", "https://github.com/oceanbase/ob-operator/blob/master/"),
]


def replace_doc_url(doc_url: str) -> str:
    """
    Replace the doc url.
    """
    for pattern, base_url in replacers:
        doc_url = re.sub(pattern, base_url, doc_url)
    return doc_url


def get_elapsed_tips(
    start_time: float,
    end_time: Optional[float] = None,
) -> str:
    end_time = end_time or time.time()
    elapsed_time = end_time - start_time
    return f" (elapsed {elapsed_time:.2f}s)"


def doc_rag_stream(
    query: str,
    chat_history: list[dict],
    suffixes: list[any] = [],
    **kwargs,
) -> Iterator[Union[str, AIMessageChunk]]:
    """
    Stream the response from the RAG model.
    """
    start_time = time.time()
    yield "Recognizing intents of the question..." + get_elapsed_tips(
        start_time, start_time
    )
    guard_res = iga.invoke_json(query)
    query_type = guard_res.get("type", "特性问题")

    if query_type == "闲聊":
        yield "Not related to OceanBase" + get_elapsed_tips(start_time)
        yield None
        for chunk in rag_agent.stream(query, chat_history, document_snippets=""):
            yield chunk
    else:
        yield "Analyzing related components..." + get_elapsed_tips(start_time)

        analyze_res = caa.invoke_json(
            query,
            background_history=chat_history,
            supported_components=supported_components,
        )
        related_comps: list[str] = analyze_res.get("components", ["observer"])

        for comp in related_comps:
            if comp not in supported_components:
                related_comps.remove(comp)

        if "observer" not in related_comps:
            related_comps.append("observer")

        print("related_comps", related_comps)
        docs = []
        rerankable = (
            os.getenv("ENABLE_RERANK", None) == "true"
            and getattr(embeddings, "rerank", None) is not None
        )
        limit = 10 if rerankable else max(3, 13 - 3 * len(related_comps))
        total_docs = []

        yield f"Embedding the question..." + get_elapsed_tips(start_time)
        query_embedded = embeddings.embed_query(query)

        for comp in related_comps:
            yield f"Searching documents related to {comp}..." + get_elapsed_tips(
                start_time
            )
            total_docs.extend(
                doc_search_by_vector(
                    query_embedded,
                    partition_names=[comp],
                    limit=limit,
                )
            )

        if rerankable and len(related_comps) > 1:
            yield "Reranking documents..." + get_elapsed_tips(start_time)
            total_docs = embeddings.rerank(query, total_docs)
            docs = total_docs[:10]
        else:
            docs = total_docs

        yield "The LLM is thinking..." + get_elapsed_tips(start_time)

        docs_content = "\n=====\n".join(
            [f"文档片段:\n\n" + chunk.page_content for i, chunk in enumerate(docs)]
        )

        ans_itr = rag_agent.stream(query, chat_history, document_snippets=docs_content)

        visited = {}
        count = 0
        buffer: str = ""
        pruned_references = []
        get_first_token = False
        whole = ""
        for chunk in ans_itr:
            whole += chunk.content
            buffer += chunk.content
            if "[" in buffer and len(buffer) < 128:
                matches = re.findall(doc_cite_pattern, buffer)
                if len(matches) == 0:
                    continue
                else:
                    sorted(matches, key=lambda x: x[0], reverse=True)
                    for m, order in matches:
                        doc = docs[int(order) - 1]
                        meta = DocumentMeta.model_validate(doc.metadata)
                        doc_name = meta.doc_name
                        doc_url = replace_doc_url(meta.doc_url)
                        idx = count + 1
                        if doc_url in visited:
                            idx = visited[doc_url]
                        else:
                            visited[doc_url] = idx
                            doc_text = f"{idx}. [{doc_name}]({doc_url})"
                            pruned_references.append(doc_text)
                            count += 1

                        ref_text = f"[[{idx}]]({doc_url})"
                        buffer = buffer.replace(m, ref_text)

            if not get_first_token:
                get_first_token = True
                yield None
            yield AIMessageChunk(content=buffer)
            buffer = ""

        print("\n\n=== RAW Output ===\n\n" + whole, "\n\n===\n\n")

        if len(buffer) > 0:
            yield AIMessageChunk(content=buffer)

        ref_tip = "根据向量相似性匹配检索到的相关文档如下:"

        if len(pruned_references) > 0:
            yield AIMessageChunk(content="\n\n" + ref_tip)

            for ref in pruned_references:
                yield AIMessageChunk(content="\n" + ref)

        elif len(docs) > 0:
            yield AIMessageChunk(content="\n\n" + ref_tip)

            visited = {}
            for doc in docs:
                meta = DocumentMeta.model_validate(doc.metadata)
                doc_name = meta.doc_name
                doc_url = replace_doc_url(meta.doc_url)
                if doc_url in visited:
                    continue
                visited[doc_url] = True
                count = len(visited)
                doc_text = f"{count}. [{doc_name}]({doc_url})"
                yield AIMessageChunk(content="\n" + doc_text)

    for suffix in suffixes:
        yield AIMessageChunk(content=suffix + "\n")
