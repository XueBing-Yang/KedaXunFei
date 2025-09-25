from typing import List, Dict, Optional, Union
from typing_extensions import TypedDict, Annotated, Literal
import operator

class DTState(TypedDict, total=False):
    # 输入
    messages: Union[List[Dict], str]
    position: str
    reflection_times: int
    background_info: str
    
    # 过程
    origin_query: str
    query: str
    reflection_idx: int
    all_queries: Annotated[List[str], operator.add]         # append 合并
    all_information: Annotated[List[str], operator.add]     # append 合并
    all_meta_information: Annotated[List[str], operator.add] # append
    create_question_answer:Annotated[List[Dict], operator.add] # 生成子问题
    question_answer:Annotated[List[Dict], operator.add] # 子问题和答案
    loop:Annotated[List[Dict], operator.add] # 反思
    summarize:Annotated[List[Dict], operator.add] #总结

    # 产出
    final_answer: str
    final_output: str

    # 元数据
    current_id: str
    current_time: str