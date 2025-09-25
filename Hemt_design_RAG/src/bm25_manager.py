"""
BM25索引管理器
用于构建、缓存和搜索BM25索引
"""
import os
import pickle
import re
from typing import List, Dict, Optional
from loguru import logger
from rank_bm25 import BM25Okapi
import json
import jieba
from tqdm import tqdm
from src.hemt_keyword import hemt_keywords
from src.hemt_config import BM25_DATA_PATH, BM25_CACHE_DIR

class BM25Manager:
    # 英文停用词列表
    ENGLISH_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
        'was', 'will', 'with', 'we', 'our', 'us', 'you', 'your', 'this', 'these',
        'those', 'they', 'them', 'their', 'his', 'her', 'him', 'she', 'or', 'but',
        'if', 'can', 'could', 'would', 'should', 'may', 'might', 'must', 'shall',
        'do', 'does', 'did', 'have', 'had', 'having', 'am', 'were', 'being',
        'not', 'no', 'nor', 'so', 'than', 'too', 'very', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
        'same', 'then', 'where', 'why', 'how', 'what', 'when', 'who', 'which',
        'while', 'after', 'before', 'during', 'above', 'below', 'up', 'down',
        'out', 'off', 'over', 'under', 'again', 'further', 'once', 'here',
        'there', 'about', 'against', 'between', 'into', 'through', 'until',
        'because', 'while', 'during'
    }
    
    CHINESE_STOPWORDS = [
    # 1. 常见虚词
    '的', '了', '和', '是', '在', '有', '也', '又', '就', '与', '及', '或', '且', '而', '但', '却', '并', '但', '以', '及',
    '这', '那', '这些', '那些', '这个', '那个', '这样', '那样', '这里', '那里', '这么', '那么', '之', '其', '此', '该',
    
    # 2. 人称代词和疑问词
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '大家', '谁', '什么', '哪', '哪里', '怎么',
    '怎样', '为什么', '如何', '多少', '几', '是否', '可以', '能够', '会', '要', '应该', '可能', '必须',
    
    # 3. 标点符号（全角+半角）
    '，', '。', '！', '？', '、', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《', '》', '…', '——', '～',
    ',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>', '-', '_', '=', '+', '*', '/', '\\',
    
    # 4. 数词、量词
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿', '个', '些', '点', '条', '件', '次',
    '年', '月', '日', '时', '分', '秒', '号', '第',
    
    # 5. 无实义高频词
    '啊', '哦', '嗯', '呃', '呀', '啦', '吧', '吗', '呢', '嘛', '哈', '唉', '喂', '哇', '喔', '诶', '呵', '嘿', '嗨', '哼',
    '对', '对于', '关于', '作为', '通过', '根据', '按照', '因为', '所以', '但是', '虽然', '如果', '即使', '尽管', '而且',
    
    # 6. 连接词和助词
    '然后', '接着', '于是', '因此', '因而', '然而', '不过', '否则', '总之', '例如', '比如', '尤其', '特别', '其实', '事实上',
    '当然', '确实', '大概', '大约', '几乎', '完全', '非常', '十分', '相当', '比较', '稍微', '只是', '只有', '只要', '除非',
    
    # 7. 其他常见停用词
    '时候', '时间', '地方', '事情', '东西', '问题', '原因', '结果', '方法', '方式', '部分', '全部', '一切', '任何', '所有',
    '每个', '某个', '一种', '一样', '一般', '一样', '一些', '一样', '一样', '一样', '一样', '一样', '一样', '一样', '一样',
]

    def __init__(self, 
                 data_path=BM25_DATA_PATH, 
                 cache_dir=BM25_CACHE_DIR):

        self.all_hemt_cases = {}
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            self.all_hemt_cases.update(json.loads(line))
        
        self.cache_dir = cache_dir

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # BM25相关属性
        self._bm25_index = None
        self._bm25_metadata = None
        self._bm25_initialized = False
        for keyword in hemt_keywords:
            jieba.add_word(keyword)
            jieba.add_word(keyword, freq=10000, tag='n')

    def get_cache_path(self) -> tuple:
        """获取缓存文件路径"""
        index_path = os.path.join(self.cache_dir, f'bm25_index_hemt.pkl')
        metadata_path = os.path.join(self.cache_dir, f'bm25_metadata_hemt.pkl')
        return index_path, metadata_path
    
    def save_index_to_cache(self):
        """将BM25索引保存到缓存文件"""
        if self._bm25_index is None or self._bm25_metadata is None:
            logger.warning("No BM25 index to save")
            return False
        
        index_path, metadata_path = self.get_cache_path()
        
        try:
            # 保存BM25索引
            with open(index_path, 'wb') as f:
                pickle.dump(self._bm25_index, f)
            
            # 保存元数据
            with open(metadata_path, 'wb') as f:
                pickle.dump(self._bm25_metadata, f)
            
            logger.info(f"BM25索引已保存到缓存: {index_path}")
            return True
        except Exception as e:
            logger.error(f"保存BM25索引到缓存失败: {e}")
            return False
    
    def load_index_from_cache(self) -> bool:
        """从缓存文件加载BM25索引"""
        index_path, metadata_path = self.get_cache_path()
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.info("BM25缓存文件不存在，需要重新构建索引")
            return False
        
        try:
            # 加载BM25索引
            with open(index_path, 'rb') as f:
                self._bm25_index = pickle.load(f)
            
            # 加载元数据
            with open(metadata_path, 'rb') as f:
                self._bm25_metadata = pickle.load(f)
            
            self._bm25_initialized = True
            logger.info(f"BM25索引已从缓存加载: {index_path}")
            return True
        except Exception as e:
            logger.error(f"从缓存加载BM25索引失败: {e}")
            return False
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        文本预处理：分词并去除停用词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 处理后的词列表
        """
        if not text:
            return []
        
        # 转为小写
        text = text.lower()
        
        # 分词
        tokens = text.split()
        
        # 过滤停用词和长度小于2的词
        filtered_tokens = [
            token for token in tokens 
            if token not in self.ENGLISH_STOPWORDS and len(token) >= 2
        ]
        
        return filtered_tokens
    
    def preprocess_text_zh(self, text: str) -> List[str]:
        words = jieba.cut(text)  # 精确模式分词
        return [word for word in words if word not in self.CHINESE_STOPWORDS] 
    
    def build_index(self, hemt_cases: List[Dict] =None) -> bool:
        
        if hemt_cases == None:
            hemt_cases = self.all_hemt_cases

        corpus = []
        doc_metadata = []
        for paper_id, cases in tqdm(hemt_cases.items()):
            try:
                for case_name, case_data in cases.items():
                    cleaned_case_data = self.clean_and_flatten(case_data)
                
                # 使用预处理方法处理文本
                tokens = self.preprocess_text_zh(cleaned_case_data)
                case_data = {paper_id: case_data}
                if tokens:  # 只添加非空的token列表
                    corpus.append(tokens)
                    doc_metadata.append(case_data)
            except:
                pass
        if corpus:
            # 构建BM25索引
            self._bm25_index = BM25Okapi(corpus)
            self._bm25_metadata = doc_metadata
            self._bm25_initialized = True
            logger.info(f"BM25索引构建完成，包含{len(corpus)}个文档")
            
            # 自动保存到缓存
            self.save_index_to_cache()
            return True
        else:
            logger.warning("No corpus built for BM25 search")
            return False
    
    def initialize_index(self, force_rebuild: bool = False) -> bool:
        """
        初始化BM25索引
        
        Args:
            force_rebuild: 是否强制重新构建索引
            
        Returns:
            bool: 初始化是否成功
        """
        if self._bm25_initialized and not force_rebuild:
            return True
        
        # 首先尝试从缓存加载
        if not force_rebuild and self.load_index_from_cache():
            return True
        
        # 如果缓存加载失败，则从Milvus构建
        return self.build_index()
    
    def search(self, query: str, topk: int = 10) -> List[Dict]:
        """
        BM25搜索
        
        Args:
            query: 查询字符串
            topk: 返回结果数量
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        if not query:
            raise ValueError("Query cannot be None or empty")
        
        # 确保索引已初始化
        if not self.initialize_index():
            logger.warning("BM25索引初始化失败")
            return []
        
        # 检查索引是否可用
        if self._bm25_index is None or self._bm25_metadata is None:
            logger.warning("BM25 index not available")
            return []
        
        # 查询预处理：应用相同的停用词过滤
        query_tokens = self.preprocess_text_zh(query)
        # logger.info(query_tokens)
        if not query_tokens:
            logger.warning("查询经预处理后为空")
            return []
        
        # 计算BM25分数
        scores = self._bm25_index.get_scores(query_tokens)
        
        # 合并分数和元数据
        scored_results = []
        for i, (score, metadata) in enumerate(zip(scores, self._bm25_metadata)):
            scored_results.append({
                **metadata,
                "bm25_score": float(score)
            })
        
        # 按BM25分数排序
        scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)
        
        # 返回top结果
        return scored_results[:topk]
    
    def clear_cache(self):
        """清除缓存文件"""
        index_path, metadata_path = self.get_cache_path()
        
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"已删除索引缓存文件: {index_path}")
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info(f"已删除元数据缓存文件: {metadata_path}")
        except Exception as e:
            logger.error(f"删除缓存文件失败: {e}")
    
    def rebuild_index(self):
        """重新构建索引"""
        logger.info("重新构建BM25索引...")
        self._bm25_index = None
        self._bm25_metadata = None
        self._bm25_initialized = False
        self.clear_cache()
        return self.initialize_index(force_rebuild=True)

    def clean_and_flatten(self, data):
        cleaned_data = self._remove_empty_items(data)  # 先清理空值
        flattened_str = self._flatten_to_string(cleaned_data)  # 再拼接字符串
        return flattened_str

    def _remove_empty_items(self, obj):
        """
        递归删除字典或列表中所有值为空的项（None, "", {}, [], "无" 等）
        """
        if isinstance(obj, dict):
            for key in list(obj.keys()):  # 遍历字典的副本
                val = obj[key]
                val = self._remove_empty_items(val)  # 递归处理值
                if self._is_empty(val):  # 判断是否为空（包括"无"）
                    del obj[key]  # 删除空键值对
                else:
                    obj[key] = val  # 更新处理后的值
            return obj
        elif isinstance(obj, (list, tuple, set)):
            new_obj = []
            for item in obj:
                item = self._remove_empty_items(item)  # 递归处理元素
                if not self._is_empty(item):  # 保留非空元素
                    new_obj.append(item)
            return type(obj)(new_obj)  # 保持原类型
        else:
            return obj  # 非容器类型直接返回

    def _is_empty(self, val):
        """
        判断值是否为空（None, "", {}, [], "无" 等）
        """
        if val is None:
            return True
        if isinstance(val, (str, bytes)) and not val.strip():  # 空字符串或仅空白字符
            return True
        if isinstance(val, (dict, list, tuple, set)) and not val:
            return True
        if isinstance(val, str) and val == "无":  # 新增：检测"无"
            return True
        return False
    
    def _flatten_to_string(self, obj):
        if isinstance(obj, dict):
            parts = []
            for val in obj.values():
                parts.append(self._flatten_to_string(val))
            return " ".join(filter(None, parts))  # 过滤None后拼接
        elif isinstance(obj, (list, tuple, set)):
            parts = []
            for item in obj:
                parts.append(self._flatten_to_string(item))
            return " ".join(filter(None, parts))
        else:
            return str(obj) if obj is not None else ""  # 非容器类型直接转字符串

if __name__ == "__main__":
    bm25_manager = BM25Manager(data_path="data/hemt_cases_taged.json", cache_dir="data/cache_dir")
    bm25_manager.initialize_index()

    search_results = bm25_manager.search("我想使用异质结提高击穿场强")
    for search_result in search_results:
        logger.info(search_result)
        break