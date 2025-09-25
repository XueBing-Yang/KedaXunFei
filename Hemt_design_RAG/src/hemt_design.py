import os
os.environ['TMPDIR'] = 'data/jieba'
import asyncio
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Union
from typing_extensions import TypedDict, Annotated, Literal
from fastapi import Request
from loguru import logger
import random
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from src.tools import extract_json_list, strip_refs, search
from src.model import ClModel, VllmModel
from src.bm25_manager import BM25Manager
from src.state_class import DTState
from src.hemt_config import METADATA_PATH
from src.prompts.hemt_design_prompts import (
    answer_one_sub_question,
    split_query,
    summary_answer,
    check_reflection,
)
with open("metadata/by_mineruID_Complete_v4.json",'r',encoding='utf-8') as f:
    metadata_information = json.load(f)

class HemtDesign:
    def __init__(self):
        self.cl_model = VllmModel()
        self.bm25_handler = BM25Manager()
        self.graph = self._build_graph()
        self.metadata = self._load_metadata()

    async def __call__(
            self,
            messages: Optional[Union[List[Dict], str]] = None,
            background_info: str = None,
            position: str = "right",
            reflection_times: int = 3,
        ):
            inputs: DTState = {
                "messages": messages,
                "position": position,
                "reflection_times": reflection_times,
                'background_info': background_info
            }

            final_state: DTState = await self.graph.ainvoke(inputs)

            return final_state

    async def init_and_rewrite(self, state: DTState) -> Command[Literal["split_and_answer"]]:
        messages = state["messages"]
        origin_query = messages
        query = messages

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return Command(
            update={
                "origin_query": origin_query,
                "query": query,
                "reflection_idx": 0,
                "all_queries": [query],
                "all_information": [],
                "current_id": now,
                "current_time": now,
                "all_meta_information": [],
                "create_question_answer": [],
                "question_answer": [],
                "loop": [],
                "summarize": []
            },
            goto="split_and_answer",
        )

    async def split_and_answer(self, state: DTState) -> Command[Literal["reflect_loop"]]:
        split_prompt = split_query.format(question=state["query"])
        create_question_answer = []
        raw = await asyncio.to_thread(self.cl_model, split_prompt)
        sub_queries = extract_json_list(raw)
        create_question_answer.append({"Prompt":split_prompt,"answer":raw})
        question_answer = []
        all_queries = [*state["all_queries"], *sub_queries] if sub_queries else state["all_queries"]
        
        results = await asyncio.gather(*(self._answer_one_query_async(q, state["background_info"],state) for q in all_queries))
        info_lines = []
        for q, (ans, _passages,i) in zip(all_queries, results):
            if ans and "没有可参考的案例" not in ans:
                info_lines.append(f"\n子问题：{q}\n答案：{ans}\n")
                question_answer.extend(i)
        return Command(
            update={
                "all_queries": all_queries, 
                "all_information": info_lines,
                "create_question_answer":create_question_answer,
                "question_answer":question_answer
                },
            goto="reflect_loop",
        )

    async def reflect_loop(self, state: DTState) -> Command[Literal["reflect_loop", "summarize"]]:
        reflection_idx = state.get("reflection_idx", 0)
        limit = state.get("reflection_times", 5)
        context_str = "\n".join(state.get("all_information", []))
        question_answer = []
        
        need, new_queries,loop = await self._check_reflection(state["query"], context_str)
        if (not need) or (not new_queries) or (reflection_idx + 1 > limit):

            return Command(update={"reflection_idx": reflection_idx}, goto="summarize")

        new_results = await asyncio.gather(*(self._answer_one_query_async(q, state['background_info'],state) for q in new_queries))
        new_infos = []
        for q, (ans, _passages,i) in zip(new_queries, new_results):
            if ans:
                state["all_information"] = [i for i in state["all_information"] if f"子问题：{q}\n答案：" not in i]
                new_infos.append(f"子问题：{q}\n答案：{ans}")
                question_answer.extend(i)
                

        return Command(
            update={
                "reflection_idx": reflection_idx + 1,
                "all_queries": new_queries,        # Annotated[List, add] => 追加
                "all_information": new_infos,      # Annotated[List, add] => 追加
                "loop": loop,
                "question_answer":question_answer
            },
            goto="reflect_loop",
        )

    async def summarize(self, state: DTState) -> Command[Literal["emit"]]:
        context_str = "\n".join(state.get("all_information", []))
        summarizes = []
        summary_prompt = summary_answer.format(
            query=state["query"], context=context_str
        )
        
        answer = await asyncio.to_thread(self.cl_model, summary_prompt)
        answer = strip_refs(answer)
        summarizes.append({"Prompt":summary_prompt,"answer":answer})
        logger.info(state)
        return Command(update={"final_answer": answer,"summarize":summarizes}, goto="emit")

    async def emit(self, state: DTState) -> Command[Literal["__end__"]]:
        final_output = state.get("final_answer", "") or ""

        logger.info(f"最终结果：\n{'=' * 50}\n{final_output}")
        return Command(update={"final_output": final_output}, goto=END)

    async def _answer_one_query_async(self, query: str, background_info: str,state:DTState):
        bm25_results = await asyncio.to_thread(
            self.bm25_handler.search, query=query, topk=20
        )
        question_answer = []
        k = min(6,len(bm25_results))
        bm25_results = random.sample(bm25_results,k=k)
        embedding_results = search(query, 20)
        k = min(2,len(embedding_results))
        embedding_results = random.sample(embedding_results, k=k)
        if bm25_results or embedding_results:
            lines = []
            for search_result in bm25_results:
                paper_key = [k for k in search_result if k != "bm25_score"][0]
                current_metadata = self.metadata.get(paper_key, None)
                lines.append(
                            f"### [案例来源：{paper_key}]\n"
                            # f"- **案例摘要**：{current_metadata['abstract']}\n"
                            f"- **内容**：\n{search_result[paper_key]}\n\n"
                        )
                state["all_meta_information"].append({paper_key:metadata_information.get(paper_key,"")})
            for search_result_paper in embedding_results:
                current_metadata = self.metadata.get(search_result_paper['doc_id'], None)
                lines.append(
                        f"### [案例来源：{search_result_paper['doc_id']}]\n"
                        # f"- **元数据**：{current_metadata['abstract']}\n"
                        f"- **内容**：\n{search_result_paper['text']}\n\n"
                    )
                state["all_meta_information"].append({search_result_paper['doc_id']:metadata_information.get(search_result_paper['doc_id'],"")})
            cases = "\n".join(lines)

            rag_prompt = answer_one_sub_question.format(query=query,background_info=background_info, cases=cases)
            rag_answer = await asyncio.to_thread(self.cl_model, rag_prompt)
            question_answer.append({"Prompt":rag_prompt,"answer":rag_answer})
            logger.info(f"current_query = {query}\n{'-' * 50}\ncurrent_cases = {cases}\n{'-' * 50}\ncurrent_answer = {rag_answer}\n\n{'=' * 50}\n\n")
            return f"{rag_answer}", lines,question_answer
        else:
            return None, [],[]

    async def _check_reflection(self, query: str, context_or_answer: str):
        check_prompt_text = check_reflection.format(
            query=query, context=context_or_answer
        )
        loop = []
        check_result = await asyncio.to_thread(self.cl_model, check_prompt_text)
        logger.info(check_result)
        loop.append({"Prompt":check_prompt_text,"answer":check_result})
        if re.search(r"NO_NEW_QUERY", check_result):
            return False, [],loop
        new_queries = extract_json_list(check_result)
        return (bool(new_queries), new_queries,loop)

    def _build_graph(self):
        builder = StateGraph(DTState)

        builder.add_node("init_and_rewrite", self.init_and_rewrite)
        builder.add_node("split_and_answer", self.split_and_answer)
        builder.add_node("reflect_loop", self.reflect_loop)
        builder.add_node("summarize", self.summarize)
        builder.add_node("emit", self.emit)

        builder.add_edge(START, "init_and_rewrite")
        builder.add_edge("init_and_rewrite", "split_and_answer")
        builder.add_edge("split_and_answer", "reflect_loop")
        builder.add_edge("summarize", "emit")
        # emit -> END 在 emit 里用 goto=END

        return builder.compile()

    def _load_metadata(self):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)


