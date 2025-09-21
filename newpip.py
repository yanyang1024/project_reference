#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG Pipeline - 基于LazyLLM框架的PDF文档问答系统

本模块实现了一个完整的多模态RAG（检索增强生成）系统，支持：
1. PDF文档的文本和图片提取
2. 多模态内容的向量化存储
3. 基于语义相似度的检索
4. 结合上下文的智能问答

主要组件：
- MultimodalRAGPipeline: 核心管道类
- MultimodalPDFReader: PDF多模态阅读器
- LazyLLM Document: 文档存储和检索
- Milvus: 向量数据库存储

作者: AI Assistant
版本: 2.0
更新时间: 2025-01-22
"""

import os
import sys
import tempfile
import logging
from typing import List, Dict, Any, Tuple

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_rag.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.info(f"已添加路径到sys.path: {parent_dir}")

# 导入自定义模块
try:
    from my_online_build.online_web_deploy import CustomEmbeddingModule, CustomChatModule
    logger.info("成功导入自定义模块: CustomEmbeddingModule, CustomChatModule")
except ImportError as e:
    logger.error(f"导入自定义模块失败: {e}")
    raise

# 导入LazyLLM相关模块
try:
    import lazyllm
    from lazyllm.tools.rag import Retriever, SentenceSplitter, DocNode, DataType
    from lazyllm.tools.rag.store import ChromadbStore
    from lazyllm import pipeline, parallel, bind, _0, _1
    logger.info("成功导入LazyLLM相关模块")
except ImportError as e:
    logger.error(f"导入LazyLLM模块失败: {e}")
    raise

# 导入PDF阅读器
try:
    from my_online_build.newreader import MultimodalPDFReader
    logger.info("成功导入MultimodalPDFReader")
except ImportError as e:
    logger.error(f"导入MultimodalPDFReader失败: {e}")
    raise

import json
import re
from typing import List, Dict, Any, Tuple

class MultimodalRAGPipeline:
    """
    多模态RAG Pipeline核心类
    
    该类封装了完整的多模态RAG流程，包括：
    1. PDF文档解析和多模态内容提取
    2. 向量化存储和索引构建
    3. 语义检索和相关性排序
    4. 上下文增强的智能问答
    
    Attributes:
        pdf_path (str): PDF文件路径
        image_save_dir (str): 图片保存目录
        reader (MultimodalPDFReader): PDF阅读器实例
        embed (CustomEmbeddingModule): 嵌入模型实例
        llm (CustomChatModule): 语言模型实例
        documents (lazyllm.Document): LazyLLM文档对象
        image_description_map (Dict): 图片路径到描述的映射
        description_image_map (Dict): 描述到图片路径的映射
        conversation_history (List): 对话历史记录
    """
    
    def __init__(self, pdf_path: str, image_save_dir: str = './images'):
        """
        初始化多模态RAG Pipeline
        
        Args:
            pdf_path (str): PDF文件路径
            image_save_dir (str): 图片保存目录，默认为'./images'
        
        Raises:
            FileNotFoundError: 当PDF文件不存在时
            OSError: 当无法创建图片保存目录时
        """
        logger.info(f"初始化MultimodalRAGPipeline - PDF路径: {pdf_path}")
        
        # 验证PDF文件存在性
        if not os.path.exists(pdf_path):
            error_msg = f"PDF文件不存在: {pdf_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.pdf_path = pdf_path
        self.image_save_dir = os.path.abspath(image_save_dir)
        
        # 创建图片保存目录
        try:
            os.makedirs(self.image_save_dir, exist_ok=True)
            logger.info(f"图片保存目录已创建/确认: {self.image_save_dir}")
        except OSError as e:
            logger.error(f"创建图片保存目录失败: {e}")
            raise
        
        # 初始化组件
        logger.info("正在初始化RAG组件...")
        try:
            self.reader = MultimodalPDFReader(image_save_dir=image_save_dir)
            self.embed = CustomEmbeddingModule()
            self.llm = CustomChatModule()
            logger.info("RAG组件初始化成功")
        except Exception as e:
            logger.error(f"RAG组件初始化失败: {e}")
            raise
        
        # 初始化存储变量
        self.documents = None
        self.image_description_map = {}  # 图片路径 -> 描述
        self.description_image_map = {}  # 描述 -> 图片路径
        self.conversation_history = []  # 对话历史
        
        logger.info("MultimodalRAGPipeline初始化完成")
        
    def build_knowledge_base(self):
        """
        构建知识库
        
        该方法执行以下步骤：
        1. 使用MultimodalPDFReader解析PDF文档
        2. 创建LazyLLM Document对象
        3. 建立文档索引和向量存储
        4. 构建图片描述映射关系
        
        Raises:
            Exception: 当知识库构建过程中出现错误时
        """
        logger.info("开始构建知识库...")
        
        try:
            # 创建LazyLLM Document对象
            logger.info("正在创建LazyLLM Document对象...")
            self.documents = lazyllm.Document(
                dataset_path=self.pdf_path, 
                embed=self.embed, 
                manager=False
            )
            
            # 创建PDF文档组并应用阅读器
            logger.info("正在创建PDF文档组...")
            self.documents.create_node_group(name="pdf_group", transform=self.reader)
            
            # 获取图片映射关系
            logger.info("正在建立图片描述映射关系...")
            self.image_description_map = self.reader.image_description_map.copy()
            self.description_image_map = self.reader.description_image_map.copy()
            
            logger.info(f"知识库构建完成 - 包含 {len(self.image_description_map)} 张图片")
            
        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            raise
        
    def retrieve_relevant_content(self, query: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        """
        检索相关内容
        
        根据用户查询，从知识库中检索最相关的文本段落和图片描述
        
        Args:
            query (str): 用户查询文本
            top_k (int): 返回的文档数量，默认为5
            
        Returns:
            Tuple[List[str], List[str]]: (文本段落列表, 图片描述列表)
            
        Raises:
            ValueError: 当知识库未构建时
            Exception: 当检索过程中出现错误时
        """
        logger.info(f"开始检索相关内容 - 查询: {query[:50]}...")
        
        if self.documents is None:
            error_msg = "请先调用 build_knowledge_base() 构建知识库"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            # 使用LazyLLM的Retriever进行检索，指定相似度算法
            logger.info(f"正在执行语义检索 - top_k: {top_k}")
            retriever = lazyllm.Retriever(
                doc=self.documents, 
                group_name="block", 
                similarity="cosine", 
                topk=top_k
            )
            retrieved_docs = retriever(query)
            
            # 分离文本和图片描述
            text_paragraphs = []
            image_descriptions = []
            
            logger.info(f"检索到 {len(retrieved_docs)} 个文档，正在分类...")
            
            for i, doc in enumerate(retrieved_docs):
                content = doc.get_text()
                metadata = doc.metadata
                
                # 根据metadata判断内容类型
                content_type = metadata.get('type', 'unknown')
                logger.debug(f"文档 {i+1}: 类型={content_type}, 内容长度={len(content)}")
                
                if content_type == 'image':
                    image_descriptions.append(content)
                else:
                    text_paragraphs.append(content)
            
            logger.info(f"检索完成 - 文本段落: {len(text_paragraphs)}, 图片描述: {len(image_descriptions)}")
            return text_paragraphs, image_descriptions
            
        except Exception as e:
            logger.error(f"检索相关内容失败: {e}")
            raise
        
    def render_images_for_display(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为界面显示准备图片信息
        
        处理图片信息，添加显示所需的元数据，如文件大小、显示URL等
        
        Args:
            images (List[Dict[str, Any]]): 原始图片信息列表
            
        Returns:
            List[Dict[str, Any]]: 处理后的图片显示信息列表
        """
        logger.info(f"正在准备 {len(images)} 张图片的显示信息...")
        
        display_images = []
        
        for i, img in enumerate(images):
            img_path = img['path']
            
            # 检查图片文件是否存在
            if not os.path.exists(img_path):
                logger.warning(f"图片文件不存在: {img_path}")
                continue
                
            # 准备显示信息
            display_info = {
                'path': img_path,
                'filename': img['filename'],
                'description': img['description'],
                'relative_path': img['relative_path'],
                'exists': True,
                'size': self._get_file_size(img_path),
                'display_url': self._get_display_url(img_path)
            }
            
            display_images.append(display_info)
            logger.debug(f"图片 {i+1} 显示信息已准备: {display_info['filename']}")
            
        logger.info(f"图片显示信息准备完成 - 有效图片: {len(display_images)}")
        return display_images
    
    def _get_file_size(self, file_path: str) -> str:
        """
        获取文件大小的人类可读格式
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            str: 格式化的文件大小字符串
        """
        try:
            size = os.path.getsize(file_path)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        except Exception as e:
            logger.warning(f"获取文件大小失败 {file_path}: {e}")
            return "未知"
    
    def _get_display_url(self, file_path: str) -> str:
        """
        获取用于显示的URL路径
        
        Args:
            file_path (str): 文件绝对路径
            
        Returns:
            str: 相对路径格式的显示URL
        """
        try:
            rel_path = os.path.relpath(file_path, os.getcwd())
            return rel_path.replace('\\', '/')  # 统一使用正斜杠
        except Exception as e:
            logger.warning(f"生成显示URL失败 {file_path}: {e}")
            return file_path.replace('\\', '/')

    def extract_image_references_from_response(self, response: str) -> List[str]:
        """
        从LLM响应中提取图片相关的描述
        
        分析LLM生成的响应文本，识别其中可能引用的图片描述
        
        Args:
            response (str): LLM生成的响应文本
            
        Returns:
            List[str]: 相关图片描述列表
        """
        logger.info("正在从响应中提取图片引用...")
        
        relevant_descriptions = []
        
        # 遍历所有图片描述，检查是否与响应相关
        for description in self.description_image_map.keys():
            if self._is_description_relevant_to_response(description, response):
                relevant_descriptions.append(description)
                logger.debug(f"找到相关图片描述: {description[:50]}...")
                
        logger.info(f"提取到 {len(relevant_descriptions)} 个相关图片引用")
        return relevant_descriptions
    
    def _is_description_relevant_to_response(self, description: str, response: str) -> bool:
        """
        判断图片描述是否与响应相关
        
        使用关键词匹配算法判断图片描述与LLM响应的相关性
        
        Args:
            description (str): 图片描述文本
            response (str): LLM响应文本
            
        Returns:
            bool: 是否相关
        """
        # 简单的关键词匹配策略
        description_words = set(description.lower().split())
        response_words = set(response.lower().split())
        
        # 计算交集比例
        intersection = description_words.intersection(response_words)
        if len(description_words) > 0:
            similarity = len(intersection) / len(description_words)
            is_relevant = similarity > 0.3  # 阈值可调整
            
            if is_relevant:
                logger.debug(f"图片描述相关性: {similarity:.3f} (阈值: 0.3)")
            
            return is_relevant
        
        return False
    
    def get_images_for_descriptions(self, descriptions: List[str]) -> List[Dict[str, Any]]:
        """
        根据描述获取对应的图片信息
        
        Args:
            descriptions (List[str]): 图片描述列表
            
        Returns:
            List[Dict[str, Any]]: 图片信息列表
        """
        logger.info(f"正在获取 {len(descriptions)} 个描述对应的图片信息...")
        
        images = []
        for i, desc in enumerate(descriptions):
            if desc in self.description_image_map:
                img_path = self.description_image_map[desc]
                if os.path.exists(img_path):
                    images.append({
                        'path': img_path,
                        'filename': os.path.basename(img_path),
                        'description': desc,
                        'relative_path': os.path.relpath(img_path, os.getcwd())
                    })
                    logger.debug(f"图片 {i+1} 信息已获取: {os.path.basename(img_path)}")
                else:
                    logger.warning(f"图片文件不存在: {img_path}")
            else:
                logger.warning(f"未找到描述对应的图片: {desc[:50]}...")
        
        logger.info(f"图片信息获取完成 - 有效图片: {len(images)}")
        return images
    
    def query_and_response(self, query: str) -> Dict[str, Any]:
        """
        执行查询并生成响应
        
        这是Pipeline的核心方法，执行完整的RAG流程：
        1. 检索相关内容
        2. 构建上下文
        3. 生成响应
        4. 提取相关图片
        5. 记录对话历史
        
        Args:
            query (str): 用户查询文本
            
        Returns:
            Dict[str, Any]: 包含响应、图片和上下文的完整结果
            
        Raises:
            ValueError: 当知识库未构建时
            Exception: 当查询处理过程中出现错误时
        """
        logger.info(f"开始处理查询: {query}")
        
        if self.documents is None:
            error_msg = "请先调用 build_knowledge_base() 构建知识库"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # 1. 检索相关内容
            logger.info("步骤1: 检索相关内容...")
            text_paragraphs, image_descriptions = self.retrieve_relevant_content(query)
            
            # 2. 构建上下文
            logger.info("步骤2: 构建上下文...")
            context_parts = []
            if text_paragraphs:
                context_parts.extend(text_paragraphs)
                logger.info(f"添加了 {len(text_paragraphs)} 个文本段落到上下文")
            if image_descriptions:
                context_parts.extend([f"图片描述: {desc}" for desc in image_descriptions])
                logger.info(f"添加了 {len(image_descriptions)} 个图片描述到上下文")
            
            context_str = "\n\n".join(context_parts)
            logger.info(f"上下文构建完成 - 总长度: {len(context_str)} 字符")
            
            # 3. 生成响应
            logger.info("步骤3: 生成LLM响应...")
            prompt = f"""你是一个面向学术文档的问答助手。你的回答必须严格依据给定的上下文，而不是先验知识。
请在保证事实准确的前提下，给出清晰、有条理的答案。若上下文无法回答，请直接说明"无法从给定内容中确定"。

已知内容：{context_str}

问题：{query}

回答："""
            
            response = self.llm(prompt)
            logger.info(f"LLM响应生成完成 - 响应长度: {len(response)} 字符")
            
            # 4. 提取响应中相关的图片
            logger.info("步骤4: 提取相关图片...")
            relevant_image_descriptions = self.extract_image_references_from_response(response)
            relevant_images = self.get_images_for_descriptions(relevant_image_descriptions)
            
            # 5. 添加到对话历史
            logger.info("步骤5: 记录对话历史...")
            conversation_entry = {
                'query': query,
                'response': response,
                'images': relevant_images,
                'context': context_str,
                'timestamp': logger.handlers[0].formatter.formatTime(logging.LogRecord(
                    name='', level=0, pathname='', lineno=0, msg='', args=(), exc_info=None
                ))
            }
            self.conversation_history.append(conversation_entry)
            
            result = {
                'response': response,
                'images': relevant_images,
                'context': context_str
            }
            
            logger.info(f"查询处理完成 - 响应长度: {len(response)}, 相关图片: {len(relevant_images)}")
            return result
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            raise

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        获取对话历史
        
        Returns:
            List[Dict[str, Any]]: 对话历史记录的副本
        """
        logger.info(f"获取对话历史 - 共 {len(self.conversation_history)} 条记录")
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """清空对话历史"""
        logger.info("清空对话历史")
        self.conversation_history = []


# 使用示例和兼容性函数
def create_multimodal_pipeline(pdf_path: str = None, image_save_dir: str = None) -> MultimodalRAGPipeline:
    """
    创建多模态RAG Pipeline实例的工厂函数
    
    提供便捷的Pipeline创建接口，支持默认参数
    
    Args:
        pdf_path (str, optional): PDF文件路径，默认为"./test.pdf"
        image_save_dir (str, optional): 图片保存目录，默认为"./images"
        
    Returns:
        MultimodalRAGPipeline: 配置好的Pipeline实例
    """
    logger.info("创建MultimodalRAGPipeline实例...")
    
    # 如果没有提供路径，使用默认的绝对路径
    if pdf_path is None:
        pdf_path = os.path.abspath("./test.pdf")
        logger.info(f"使用默认PDF路径: {pdf_path}")
    if image_save_dir is None:
        image_save_dir = os.path.abspath("./images")
        logger.info(f"使用默认图片目录: {image_save_dir}")
    
    pipeline = MultimodalRAGPipeline(pdf_path, image_save_dir)
    logger.info("MultimodalRAGPipeline实例创建完成")
    return pipeline


# 为了保持向后兼容性，保留原有的简单接口
def query_and_response(query: str, pipeline: MultimodalRAGPipeline = None) -> str:
    """
    简单的查询接口（向后兼容）
    
    提供简化的查询接口，自动处理Pipeline的创建和初始化
    
    Args:
        query (str): 用户查询文本
        pipeline (MultimodalRAGPipeline, optional): 可选的pipeline实例
        
    Returns:
        str: 响应文本
    """
    logger.info(f"使用简化接口处理查询: {query}")
    
    if pipeline is None:
        logger.info("创建默认Pipeline实例...")
        # 使用默认配置创建pipeline
        pipeline = create_multimodal_pipeline()
        if pipeline.documents is None:
            logger.info("构建知识库...")
            pipeline.build_knowledge_base()
    
    result = pipeline.query_and_response(query)
    logger.info("简化接口查询完成")
    return result['response']


# -----------------------------------------------------
# Configuration - 系统配置模块
# -----------------------------------------------------

def get_cache_path():
    """
    获取缓存路径
    
    Returns:
        str: 缓存目录路径
    """
    cache_path = os.path.join(lazyllm.config['home'], 'rag_for_qa')
    logger.debug(f"缓存路径: {cache_path}")
    return cache_path

def get_image_path():
    """
    获取图片存储路径
    
    Returns:
        str: 图片存储目录路径
    """
    image_path = os.path.join(get_cache_path(), "images")
    logger.debug(f"图片路径: {image_path}")
    return image_path

class TmpDir:
    """
    临时目录管理类
    
    负责管理RAG系统所需的各种临时目录和文件路径
    
    Attributes:
        root_dir (str): 根目录路径
        rag_dir (str): RAG数据目录路径
        store_file (str): Milvus数据库文件路径
        image_path (str): 图片存储路径
    """
    
    def __init__(self):
        """初始化临时目录管理器"""
        logger.info("初始化临时目录管理器...")
        
        # 设置根目录
        self.root_dir = os.path.expanduser(os.path.join(lazyllm.config['home'], 'rag_for_qa'))
        self.rag_dir = os.path.join(self.root_dir, "rag_master")
        
        # 创建RAG目录
        try:
            os.makedirs(self.rag_dir, exist_ok=True)
            logger.info(f"RAG目录已创建: {self.rag_dir}")
        except OSError as e:
            logger.error(f"创建RAG目录失败: {e}")
            raise
        
        # 设置存储文件路径
        self.store_file = os.path.join(self.root_dir, "milvus.db")
        logger.info(f"Milvus数据库文件路径: {self.store_file}")
        
        # 设置图片路径
        self.image_path = get_image_path()
        try:
            os.makedirs(self.image_path, exist_ok=True)
            logger.info(f"图片目录已创建: {self.image_path}")
        except OSError as e:
            logger.error(f"创建图片目录失败: {e}")
            raise

    def cleanup(self):
        """
        清理临时文件和目录
        
        删除Milvus数据库文件和所有图片文件
        """
        logger.info("开始清理临时文件...")
        
        # 删除Milvus数据库文件
        if os.path.isfile(self.store_file):
            try:
                os.remove(self.store_file)
                logger.info(f"已删除Milvus数据库文件: {self.store_file}")
            except OSError as e:
                logger.error(f"删除Milvus数据库文件失败: {e}")
        
        # 删除图片文件
        if os.path.exists(self.image_path):
            try:
                for filename in os.listdir(self.image_path):
                    filepath = os.path.join(self.image_path, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                        logger.debug(f"已删除图片文件: {filename}")
                logger.info("图片文件清理完成")
            except OSError as e:
                logger.error(f"清理图片文件失败: {e}")

# 创建全局临时目录管理器实例
tmp_dir = TmpDir()

# 本地存储配置 - Milvus向量数据库配置
milvus_store_conf = {
    "type": "milvus",
    "kwargs": {
        'uri': tmp_dir.store_file,  # 使用本地文件作为Milvus存储
        'index_kwargs': {
            'index_type': 'HNSW',      # 使用HNSW索引算法
            'metric_type': "COSINE",   # 使用余弦相似度
        }
    },
}
logger.info(f"Milvus存储配置已设置: {milvus_store_conf}")

# 文档字段配置 - 定义文档的元数据字段
from lazyllm.tools.rag import DocField, DataType
doc_fields = {
    'comment': DocField(data_type=DataType.VARCHAR, max_size=65535, default_value=' '),
    'signature': DocField(data_type=DataType.VARCHAR, max_size=32, default_value=' '),
}
logger.info(f"文档字段配置已设置: {list(doc_fields.keys())}")

# 系统路径配置
PDF_PATH = "E:\\890\\111\\pdfqa\\Algorithm.pdf"
BASE_DIR = 'E:\\890\\111\\pdfqa\\'
IMAGE_DIR = tmp_dir.image_path
TEXT_STORE_DIR = os.path.join(BASE_DIR, "chroma_text_db")
DESC_STORE_DIR = os.path.join(BASE_DIR, "chroma_desc_db")

# Ollama服务配置
OLLAMA_BASE = "http://127.0.0.1:11434/api/"
OLLAMA_EMBED_URL = OLLAMA_BASE + "embed"
OLLAMA_MODEL = "gemma3"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
SERVER_PORT = 23480

logger.info(f"系统配置加载完成:")
logger.info(f"  - PDF路径: {PDF_PATH}")
logger.info(f"  - 图片目录: {IMAGE_DIR}")
logger.info(f"  - Ollama服务: {OLLAMA_BASE}")
logger.info(f"  - 服务端口: {SERVER_PORT}")

# 系统提示词模板
PROMPT = (
    "你是一个面向学术文档的问答助手。你的回答必须严格依据给定的上下文，而不是先验知识。"
    "请在保证事实准确的前提下，给出清晰、有条理的答案。若上下文无法回答，请直接说明\"无法从给定内容中确定\"。\n"
    "已知内容：{context_str}\n"
    "问题：{query}\n"
    "回答："
)

# -----------------------------------------------------
# Utilities - 工具函数模块
# -----------------------------------------------------

def _dedup_and_format(nodes: List[DocNode], *, max_chars: int = 3500, text_weight: float = 0.6, desc_weight: float = 0.4) -> Dict[str, Any]:
    """
    去重并格式化文档节点
    
    对检索到的文档节点进行去重处理，并根据权重进行排序和格式化
    
    Args:
        nodes (List[DocNode]): 文档节点列表
        max_chars (int): 最大字符数限制，默认3500
        text_weight (float): 文本权重，默认0.6
        desc_weight (float): 描述权重，默认0.4
        
    Returns:
        Dict[str, Any]: 包含格式化上下文字符串和节点列表的字典
    """
    logger.info(f"开始去重和格式化 {len(nodes)} 个文档节点...")
    
    seen_text = set()
    uniq_nodes = []
    
    # 去重处理
    for i, n in enumerate(nodes):
        t = (n.text or '').strip()
        if not t:
            logger.debug(f"节点 {i+1}: 跳过空文本")
            continue
            
        key = t.replace("\n", " ").strip()
        if key in seen_text:
            logger.debug(f"节点 {i+1}: 跳过重复文本")
            continue
            
        seen_text.add(key)
        
        # 根据类型设置权重
        tp = (n.metadata or {}).get('type')
        score = getattr(n, 'score', 0.0) or 0.0
        
        if tp == 'text':
            score *= text_weight
        elif tp == 'image_description':
            score *= desc_weight
            
        n.score = score
        uniq_nodes.append(n)
        logger.debug(f"节点 {i+1}: 类型={tp}, 权重后得分={score:.3f}")
    
    # 按得分排序
    uniq_nodes.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
    logger.info(f"去重完成，保留 {len(uniq_nodes)} 个唯一节点")

    # 格式化为上下文字符串
    parts = []
    total = 0
    
    for i, n in enumerate(uniq_nodes):
        tag = (n.metadata or {}).get('type', 'text')
        seg = f"[{tag}] {n.text.strip()}"
        
        if total + len(seg) > max_chars:
            logger.info(f"达到字符数限制 {max_chars}，停止添加节点")
            break
            
        parts.append(seg)
        total += len(seg)
        logger.debug(f"添加节点 {i+1}: 类型={tag}, 长度={len(seg)}, 累计={total}")
    
    result = {"context_str": "\n\n".join(parts), "nodes": uniq_nodes}
    logger.info(f"格式化完成 - 最终上下文长度: {len(result['context_str'])} 字符")
    
    return result


def _append_related_images(text: str, desc_docs: List[DocNode]) -> str:
    """
    在文本后追加相关图片的Markdown引用
    
    Args:
        text (str): 原始文本
        desc_docs (List[DocNode]): 图片描述文档节点列表
        
    Returns:
        str: 追加了图片引用的文本
    """
    logger.info(f"开始追加相关图片引用 - 图片描述文档: {len(desc_docs)}")
    
    if not desc_docs:
        logger.info("没有图片描述文档，返回原文本")
        return text

    lines = ["\n\n相关图片："]
    used = set()
    
    # 显示前3张相关图片
    for i, n in enumerate(desc_docs[:3]):
        p = (n.metadata or {}).get('image_path')
        if not p or p in used:
            logger.debug(f"图片 {i+1}: 跳过 - 路径为空或已使用")
            continue
            
        used.add(p)
        name = os.path.basename(p)
        
        # WebModule with static_paths supports file= absolute path
        markdown_ref = f"![{name}](file={p})"
        lines.append(markdown_ref)
        logger.debug(f"图片 {i+1}: 添加引用 - {name}")
    
    if len(lines) == 1:
        logger.info("没有有效的图片引用，返回原文本")
        return text
    
    result = text + "\n" + "\n".join(lines)
    logger.info(f"图片引用追加完成 - 添加了 {len(lines)-1} 张图片")
    return result


# -----------------------------------------------------
# Build knowledge stores - 知识库构建模块
# (1.single doc)-(text + image description) 
# (2.multi doc)-(doc manual)-(single doc) - (text + image description)
# -----------------------------------------------------

def build_knowledge(pdf_path: str):
    """
    从PDF构建知识库的核心函数
    
    该函数执行完整的知识库构建流程：
    1. 验证PDF文件存在性
    2. 使用MultimodalPDFReader解析PDF
    3. 提取文本和图片描述文档
    4. 去重处理
    5. 创建LazyLLM Document对象
    6. 配置向量存储和检索
    
    Args:
        pdf_path (str): PDF文件路径
        
    Returns:
        Dict: 包含documents、text_docs、desc_docs的字典
        
    Raises:
        FileNotFoundError: 当PDF文件不存在时
        Exception: 当知识库构建过程中出现错误时
    """
    logger.info(f"开始构建知识库 - PDF路径: {pdf_path}")
    
    # 验证PDF文件存在性
    if not os.path.exists(pdf_path):
        error_msg = f"PDF文件不存在: {pdf_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # 确保目录存在
        logger.info("确保必要目录存在...")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        
        # 使用多模态PDF阅读器
        logger.info("初始化MultimodalPDFReader...")
        reader = MultimodalPDFReader(image_save_dir=IMAGE_DIR)
        
        # 加载和解析文档
        logger.info("开始解析PDF文档...")
        docs = reader._load_data(pdf_path)
        
        # 分离文本和图片描述文档
        logger.info("分离文本和图片描述文档...")
        text_docs = [d for d in docs if (d.metadata or {}).get('type') == 'text' and (d.text or '').strip()]
        desc_docs = [d for d in docs if (d.metadata or {}).get('type') == 'image_description' and (d.text or '').strip()]
        
        logger.info(f"文档解析完成 - 文本文档: {len(text_docs)}, 图片描述文档: {len(desc_docs)}")
        
        # 去重处理函数
        def dedup_by_text(docs):
            """按文本内容去重"""
            logger.info(f"开始去重处理 - 输入文档数: {len(docs)}")
            s = set()
            out = []
            
            for i, it in enumerate(docs):
                t = it.text.strip()
                if not t or t in s:
                    logger.debug(f"文档 {i+1}: 跳过 - 空文本或重复")
                    continue
                s.add(t)
                out.append(it)
                logger.debug(f"文档 {i+1}: 保留 - 文本长度: {len(t)}")
            
            logger.info(f"去重完成 - 输出文档数: {len(out)}")
            return out

        # 执行去重
        logger.info("对文本文档执行去重...")
        text_docs = dedup_by_text(text_docs)
        
        logger.info("对图片描述文档执行去重...")
        desc_docs = dedup_by_text(desc_docs)

        # 创建LazyLLM Document使用官方配置
        logger.info("创建嵌入模型...")
        embedder = lazyllm.TrainableModule("bge-large-zh-v1.5")
        
        # 复制PDF文件到数据集目录以确保被加载
        logger.info("准备PDF文件到数据集目录...")
        import shutil
        pdf_filename = os.path.basename(pdf_path)
        target_pdf_path = os.path.join(tmp_dir.rag_dir, pdf_filename)
        
        if not os.path.exists(target_pdf_path):
            os.makedirs(tmp_dir.rag_dir, exist_ok=True)
            shutil.copy2(pdf_path, target_pdf_path)
            logger.info(f"PDF文件已复制到: {target_pdf_path}")
        else:
            logger.info(f"PDF文件已存在: {target_pdf_path}")
        
        # 创建具有适当数据集路径和配置的文档对象
        logger.info("创建LazyLLM Document对象...")
        documents = lazyllm.Document(
            dataset_path=tmp_dir.rag_dir,
            embed=embedder,
            manager=False,
            store_conf=milvus_store_conf,
            doc_fields=doc_fields
        )
        
        # 为PDF文件添加阅读器
        logger.info("配置PDF阅读器...")
        documents.add_reader("*.pdf", reader)
        
        # 为文本块创建节点组
        logger.info("创建文本块节点组...")
        documents.create_node_group(name="block", transform=lambda s: s.split("\n") if s else '')
        
        result = {
            'documents': documents,
            'text_docs': text_docs,
            'desc_docs': desc_docs
        }
        
        logger.info(f"知识库构建成功 - 文本文档: {len(text_docs)}, 图片描述文档: {len(desc_docs)}")
        return result
        
    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        raise


def main():
    """
    主函数 - 启动多模态RAG系统
    
    该函数执行以下步骤：
    1. 配置PDF路径
    2. 构建知识库
    3. 创建检索器
    4. 构建处理管道
    5. 启动Web界面
    """
    logger.info("=" * 60)
    logger.info("启动多模态RAG系统")
    logger.info("=" * 60)
    
    # 配置PDF路径
    pdf_path = r"e:\890\111\LazyLLM\my_online_build\data\deepseek-r1.pdf"
    logger.info(f"目标PDF文件: {pdf_path}")
    
    # 验证PDF文件存在
    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        return
    
    try:
        # 构建知识库
        logger.info("步骤1: 构建知识库...")
        result = build_knowledge(pdf_path)
        documents = result['documents']
        desc_docs = result['desc_docs']
        
        # 创建检索器配置
        logger.info("步骤2: 创建检索器...")
        retriever = lazyllm.Retriever(
            doc=documents,
            group_name="block",
            topk=3,
            similarity="cosine"
        )
        logger.info("检索器创建完成 - 配置: topk=3, similarity=cosine")
        
        # 创建处理管道
        logger.info("步骤3: 构建处理管道...")
        with lazyllm.pipeline() as ppl:
            # 检索组件
            ppl.retriever = retriever
            logger.info("  - 检索器组件已添加")
            
            # 合并组件 - 将检索结果和查询合并
            ppl.merge = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
            logger.info("  - 合并组件已添加")
            
            # 格式化组件 - 格式化上下文和查询
            ppl.formatter = (lambda context_str, query: f"Context: {context_str}\n\nQuestion: {query}") | bind(query=ppl.input)
            logger.info("  - 格式化组件已添加")
            
            # LLM组件 - 使用InternLM2模型生成回答
            ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
                lazyllm.ChatPrompter(instruction=PROMPT, extra_keys=['context_str']))
            logger.info("  - LLM组件已添加 (internlm2-chat-7b)")
            
            # 后处理组件
            ppl.post = lambda x: x
            logger.info("  - 后处理组件已添加")
        
        logger.info("处理管道构建完成")
        
        # 启动Web界面
        logger.info("步骤4: 启动Web界面...")
        web_port = 23456
        logger.info(f"Web服务将在端口 {web_port} 启动")
        logger.info(f"访问地址: http://localhost:{web_port}")
        
        # 启动WebModule
        web_module = lazyllm.WebModule(ppl, port=web_port)
        logger.info("Web模块已创建，正在启动服务...")
        
        web_module.start().wait()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        raise
    finally:
        logger.info("多模态RAG系统已关闭")

 
if __name__ == '__main__':
    main()