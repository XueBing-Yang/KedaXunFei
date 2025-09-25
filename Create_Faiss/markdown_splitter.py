from genericpath import isdir
import json
import re
import os
import base64
from typing import List, Dict, NamedTuple, Optional, Tuple
from loguru import logger
from bs4 import BeautifulSoup

# 定义一个不可变的数据结构，用于存储分块结果
class Chunk(NamedTuple):
    content: str           # 分块后的文本或其他模态内容
    metadata: dict = {}    # 附带的元数据（如来源文件、标题信息等）
    modal: str = "text"    # 内容类型（例如 "text", "table", "image"）

class MarkdownSplitter:
    """Markdown文档分块处理器，用于对Markdown文档进行预处理、分块以及后续整合。"""
    
    # 正则表达式，用于检测中文和英文句子的边界
    SENTENCE_BOUNDARY = re.compile(
        r'([。！？]|[.!?])(?=\s+[A-Z]|\s+[^a-z]|\s*$)',  # 英文标点需后接大写字母或行尾
        flags=re.MULTILINE
    )
    HEADER_RE = re.compile(r'^(#+)\s+(.*)', flags=re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 1200,          # 单个分块的最大字符数
        chunk_overlap: int = 200,        # 分块之间保留的上下文字符数
        headers_to_split_on: List[Tuple[str, str]] = [
            ('#', 'Header 1'),
            ('##', 'Header 2'),
            ('###', 'Header 3')
        ]
    ):
        self.chunk_size = chunk_size
        # 重叠字符数不能超过 chunk_size 的一半
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        # 根据标题标记长度从大到小排序，确保先匹配最长的标题标记
        self.headers_to_split_on = sorted(
            headers_to_split_on,
            key=lambda x: len(x[0]),
            reverse=True
        )
    
    def _create_metadata(self, filepath: str, metadata: dict) -> dict:
        """
        创建基础元数据，设置文档来源和类型，同时合并用户传入的非空元数据项。
        """
        return {
            "source": filepath,
            **{k: v for k, v in metadata.items() if v is not None}
        }
        
    def _remove_references(self, text: str) -> str:
        """
        去掉参考文献部分：
        - 支持中英文参考文献标题识别；
        - 支持编号+年份格式识别；
    
        返回:
            str: 移除参考文献后的文本。
        """
        reference_keywords = [
            r"references", r"reference", r"bibliography", r"acknowledgements", r"acknowledgement", r"appendix", r"works cited",
            r"参考文献", r"致谢", r"附录"
        ]
    
        keyword_pattern = re.compile(r"(?i)^\s*(#+\s*)?(" + "|".join(reference_keywords) + r")\s*[:：\-]?\s*$")
        reference_number_pattern = re.compile(r"^\s*(\[\d+\]|\(\d+\)|\d+\.)\s+")
        year_pattern = re.compile(r"\b(19|20)\d{2}\b")
        inline_citation_pattern = re.compile(r"\[\d+\].*?\b(19|20)\d{2}\b")  # 例如: [1] XXX. 2001.

        lines = text.splitlines()
        reference_start_index = None

        for i, line in enumerate(lines):
            if keyword_pattern.match(line):
                reference_start_index = i
                break
            elif reference_number_pattern.match(line) and year_pattern.search(line):
                reference_start_index = i
                break

        if reference_start_index is not None:
            text = "\n".join(lines[:reference_start_index])
        else:
            text = text

        # 去除散落的文末参考文献（如：[1] ... 1999）
        text = re.sub(inline_citation_pattern, '', text)

        return text.strip()

    def _encode_image_base64(self, image_path: str) -> str:
        """
        将图片文件编码为Base64字符串。
        """
        with open(image_path, "rb") as image_file:
            base64_bytes = base64.b64encode(image_file.read())
            base64_string = base64_bytes.decode('utf-8')
        return base64_string 

    def _extract_images(self, text: str, metadata: dict) -> List[Chunk]:
        """
        提取图片块并替换为占位符 [[Image_{i}]]，i从1开始计数。
        """
        images = []

        # ---------- 1. 提取 Markdown 图片 ----------
        def replace_md_img(match):
            path = match.group(1)
            idx = len(images) + 1
            clean_path = path.split('#')[0].split('?')[0].strip()
            image_path = os.path.join(metadata['source'], clean_path)
            images.append(Chunk(
                content=self._encode_image_base64(image_path),
                metadata={
                    **metadata,
                    "image_path": image_path,
                    "id": f"Image_{idx}"
                },
                modal="image"
            ))
            return f"[[Image_{idx}]]"

        text = re.sub(r'!\[.*?\]\((.*?)\)', replace_md_img, text)

        # ---------- 2. 提取 HTML 图片 ----------
        def replace_html_img(match):
            path = match.group(1)
            idx = len(images) + 1
            clean_path = path.split('#')[0].split('?')[0].strip()
            image_path = os.path.join(metadata['source'], clean_path)
            images.append(Chunk(
                content=self._encode_image_base64(image_path),
                metadata={
                    **metadata,
                    "image_path": image_path,
                    "id": f"Image_{idx}"
                },
                modal="image"
            ))
            return f"[[Image_{idx}]]"

        text = re.sub(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', replace_html_img, text, flags=re.IGNORECASE)

        return images, text
    
    def _extract_tables(self, text: str, metadata: dict) -> Tuple[List[Chunk], str]:
        """
        提取表格块并替换为占位符 [[Table_{i}]]，i从1开始计数。
        """
        tables = []

        # ---------- 1. 提取 Markdown 表格 ----------
        markdown_pattern = re.compile(
        r"""
        (                           # 捕获整个表格
          ^\|.*\|\s*\n             # 表头行
          ^\|[-\s|:]+?\|\s*\n      # 分隔符行，要求只有 -、空格、| 或 :（用于对齐）
          (?:^\|.*\|\s*(?:\n|$))*   # 可选的数据行，每行以 | 开始
        )
        """,
        re.MULTILINE | re.VERBOSE
    )
        
        # 替换 Markdown 表格为占位符
        def md_table_replacer(match):
            idx = len(tables) + 1
            tables.append(Chunk(
                content=match.group(0).strip(),
                modal="table",
                metadata={
                    **metadata,
                    "id": f"Table_{idx}"
                    }
            ))
            return f"[[Table_{idx}]]"

        text = markdown_pattern.sub(md_table_replacer, text)

        # ---------- 2. 提取 HTML 表格 ----------
        soup = BeautifulSoup(text, "html.parser")
        for table in soup.find_all("table"):
            idx = len(tables) + 1
            full_table_html = str(table)
            tables.append(Chunk(
                content=full_table_html,
                modal="table",
                metadata={
                    **metadata,
                    "idx": f"Table_{idx}"
                    }
            ))
            # 替换为占位符
            placeholder = soup.new_string(f"[[Table_{idx}]]")
            table.replace_with(placeholder)
        
        # 获取删除表格后的文本内容
        text = soup.get_text()

        return tables, text 

    def _split_by_headers(self, text: str, metadata: dict) -> List[Chunk]:
        """将文本按最顶层标题进行段落化，每段内容包含标题与其后续文本"""
        lines = text.splitlines(keepends=True)
        sections = []
        current_lines = []
        current_meta = dict(metadata)
        for line in lines:
            m = self.HEADER_RE.match(line)
            if m:
                if current_lines:
                    sections.append(Chunk(''.join(current_lines).strip(), current_meta, 'text'))
                current_lines = [line]
                level = len(m.group(1)); title = m.group(2).strip()
                current_meta = {**metadata, f'header_{level}': title}
            else:
                current_lines.append(line)
        if current_lines:
            sections.append(Chunk(''.join(current_lines).strip(), current_meta, 'text'))
        # 过滤掉孤立的标题段
        filtered = []
        for sec in sections:
            # 如果内容仅是一行标题，且后面无正文，则跳过
            lines = sec.content.splitlines()
            if len(lines) == 1 and self.HEADER_RE.match(lines[0]):
                continue
            filtered.append(sec)
        return filtered

    def _split_text(self, text: str, metadata: dict) -> List[Chunk]:
        parts = [p for p in self.SENTENCE_BOUNDARY.split(text) if p]
        # 重组完整句子
        sents = []
        for i in range(0, len(parts), 2):
            seg = parts[i] + (parts[i+1] if i+1 < len(parts) else '')
            sents.append(seg)
        # 按句子和长度拆分
        chunks = []
        buf = []
        total_len = 0
        for sent in sents:
            sl = len(sent)
            if buf and total_len + sl > self.chunk_size:
                chunk_txt = ''.join(buf).strip()
                chunks.append(Chunk(chunk_txt, metadata, 'text'))
                # 重叠部分按句子，不使用字符截断
                overlap_sent_count = max(1, int(self.chunk_overlap / (self.chunk_size/len(buf))))
                buf = buf[-overlap_sent_count:]
                total_len = sum(len(s) for s in buf)
            buf.append(sent)
            total_len += sl
        if buf:
            final = ''.join(buf).strip()
            chunks.append(Chunk(final, metadata, 'text'))
        return chunks
    
    
    def split(self, text: str, filepath: str = '', metadata: dict = None) -> List[Chunk]:
        metadata = metadata or {}
        base_meta = {"source": filepath, **metadata}

        text = self._remove_references(text)
        # images, text = self._extract_images(text, base_meta)
        tables, text = self._extract_tables(text, base_meta)

        chunks: List[Chunk] = []
        # 先处理文本块
        text_blk = Chunk(text, base_meta, 'text')
        sections = self._split_by_headers(text_blk.content, text_blk.metadata)
        for sec in sections:
            chunks.extend(self._split_text(sec.content, sec.metadata))
        # 再附加表格与图片块
        chunks.extend(tables)
        # chunks.extend(images)

        # 过滤掉重复的块，去掉空块
        unique_chunks = []
        seen_contents = set()
        for chunk in chunks:
            if chunk.content.strip() and chunk.content not in seen_contents:
                unique_chunks.append(chunk)
                seen_contents.add(chunk.content)
    
        return unique_chunks   
    
    def split_markdown(self, text: str, filepath: str = '', metadata: dict = None, modal: str = 'text') -> List[str]:
        """获取指定模态的数据"""
        chunks = self.split(text=text,filepath=filepath, metadata=metadata)
        chunks_list = [c.content for c in chunks if c.modal == modal]

        return chunks_list

def get_paper_list(base_path:str)->list:
    paper_list = []
    for file in os.listdir(base_path):
        extracted_path = os.path.join(base_path,file)
        if not os.path.isdir(extracted_path):
            continue
        for mid in os.listdir(extracted_path):
            auto_path = os.path.join(extracted_path,mid,"auto")
            md_path = None
            for f in os.listdir(auto_path):
                if f.endswith(".md"):
                    md_path = os.path.join(auto_path,f)
                    break
            if not md_path:
                continue
            with open(md_path,'r',encoding='utf-8') as md:
                md_content = md.read()
                result = {
                    "paper":md_content,
                    "mineruID":mid
                }
            paper_list.append(result)
    return paper_list

# 使用示例
if __name__ == "__main__":
    hemt_case = "D:/Prompt_V2/hemt_cases_final_V6.jsonl"
    paper_path = "D:/Prompt_V2/J-6718_el_Prompt.json"
    mids = []
    with open(hemt_case, "r", encoding="utf-8") as f:
        hemts = [json.loads(s) for s in f.readlines()]
    # 获取mineruID
    for hemt in hemts:
        for mid,case in hemt.items():
            mids.append(mid)
    # 根据mineruID获取paper(不全)
    # with open(paper_path,'r',encoding="utf-8") as f:
    #     paper_list = []
    #     papers = [json.loads(s) for s in f.readlines()]
    #     for mid in mids:
    #         for paper in papers:
    #             if mid == paper["mineruID"]:
    #                 result = {
    #                     "paper":paper["query"].split("文献如下：\n")[1].split("## 器件结构如下")[0],
    #                     "mineruID":paper["mineruID"]
    #                 }
    #                 paper_list.append(result)
    paper_list = get_paper_list("D:/Mineru_Process/pdf_results")
    # 处理paper,并进行写入
    splitter = MarkdownSplitter(chunk_size=2048, chunk_overlap=200)
    for paper_dict in paper_list:
        md_content = paper_dict["paper"]
        input_path = "default"
        chunks = splitter.split(text=md_content, filepath=os.path.dirname(input_path))
        with open(f'Embedding/chunks/{paper_dict["mineruID"]}.txt','w',encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"Chunks(md_content):\n")
                logger.info(chunk.metadata)
                f.write(chunk.content + "\n")
    # # text_chunks = splitter.split_markdown(md_content)
    # # img_chunks = splitter.split_markdown(md_content, modal="image")
    # # table_chunks = splitter.split_markdown(md_content, modal="table")

    

    # with open('Embedding/10.1063@1.5102085.txt', 'w', encoding='utf-8') as f:
    #     for i, chunk in enumerate(chunks, 1):
    #         f.write(f"Chunk {i}({len(chunk.content)} chars):\n")
    #         logger.info(chunk.metadata)
    #         f.write(chunk.content + "\n")
    #         f.write("-" * 80 + "\n")

    #     # for i, chunk in enumerate(text_chunks, 1):
    #     #     f.write(f"Chunk {i}({len(chunk)} chars):\n")
    #     #     f.write(chunk + "\n")
    #     #     f.write("-" * 80 + "\n")
    #     # for i, chunk in enumerate(img_chunks, 1):
    #     #     f.write(f"Image Chunk {i}({len(chunk)} chars):\n")
    #     #     f.write(chunk + "\n")
    #     #     f.write("-" * 80 + "\n")
    #     # for i, chunk in enumerate(table_chunks, 1):
    #     #     f.write(f"Table Chunk {i}({len(chunk)} chars):\n")
    #     #     f.write(chunk + "\n")
    #     #     f.write("-" * 80 + "\n")
    
