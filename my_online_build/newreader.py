#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态PDF阅读器模块

本模块提供了一个增强的PDF阅读器，支持：
1. 文本内容提取和处理
2. 图片提取和智能合并
3. 图片描述自动生成
4. 上下文感知的内容分析
5. 多模态内容索引和映射

主要功能：
- 基于PyMuPDF的高质量PDF解析
- 智能图片检测和提取
- 相邻图片自动合并
- 基于上下文的图片描述生成
- 完整的元数据管理
- 错误处理和性能监控

技术特性：
- 支持高DPI图片提取（300 DPI）
- 智能文本块检测和分析
- 上下文窗口提取
- 图片-描述双向映射
- 详细的调试信息输出

作者: AI Assistant
版本: 2.0
更新时间: 2025-01-22
"""

# 标准库导入
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
import traceback

# 第三方库导入
import fitz  # PyMuPDF - PDF处理核心库
from PIL import Image  # 图片处理库
import requests
from urllib.parse import urljoin

# LazyLLM框架导入
import lazyllm
from lazyllm.tools.rag.readers import ReaderBase
from lazyllm.tools.rag import DocNode, DocField, DataType, NodeTransform
from lazyllm.module import OnlineChatModuleBase, OnlineEmbeddingModuleBase

# 自定义模块导入
try:
    from my_online_build.online_web_deploy import CustomChatModule
    logging.info("成功导入CustomChatModule")
except ImportError as e:
    logging.error(f"导入CustomChatModule失败: {e}")
    # 提供备用方案
    CustomChatModule = None

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_pdf_reader.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultimodalPDFReader(ReaderBase):
    """
    多模态PDF阅读器类
    
    这是一个高级PDF阅读器，能够同时处理文本和图片内容，
    并为图片生成智能描述。主要特性包括：
    
    1. 文本提取：
       - 基于PyMuPDF的高质量文本提取
       - 保留文本块的位置信息
       - 支持复杂布局的PDF文档
    
    2. 图片处理：
       - 高分辨率图片提取（300 DPI）
       - 智能图片合并算法
       - 上下文感知的图片描述生成
       - 完整的图片元数据管理
    
    3. 内容索引：
       - 图片路径到描述的映射
       - 描述到图片路径的反向映射
       - 支持快速内容检索
    
    4. 错误处理：
       - 完善的异常处理机制
       - 详细的错误日志记录
       - 优雅的降级处理
    
    Attributes:
        image_save_dir (str): 图片保存目录的绝对路径
        llm: 用于生成图片描述的语言模型实例
        image_description_map (Dict[str, str]): 图片路径到描述的映射
        description_image_map (Dict[str, str]): 描述到图片路径的反向映射
        processing_stats (Dict): 处理统计信息
    """
    
    def __init__(self, image_save_dir: str = './images', llm: Any = None):
        """
        初始化多模态PDF阅读器
        
        Args:
            image_save_dir (str): 图片保存目录，默认为'./images'
            llm (Any): 用于生成图片描述的语言模型，如果为None则使用默认模型
            
        Raises:
            OSError: 当无法创建图片保存目录时
            ImportError: 当无法导入必要的依赖时
        """
        logger.info("初始化MultimodalPDFReader...")
        super().__init__()
        
        # 确保图片保存目录存在
        self.image_save_dir = os.path.abspath(image_save_dir)
        try:
            os.makedirs(self.image_save_dir, exist_ok=True)
            logger.info(f"图片保存目录已创建/确认: {self.image_save_dir}")
        except OSError as e:
            logger.error(f"创建图片保存目录失败: {e}")
            raise OSError(f"无法创建图片保存目录 {self.image_save_dir}: {e}")
        
        # 初始化LLM
        if llm is None:
            try:
                if CustomChatModule is not None:
                    self.llm = CustomChatModule(
                        base_url='http://127.0.0.1:11434/api/', 
                        model='gemma3'
                    )
                    logger.info("使用默认CustomChatModule初始化LLM")
                else:
                    logger.warning("CustomChatModule不可用，使用备用LLM")
                    # 这里可以添加备用LLM的初始化逻辑
                    self.llm = None
            except Exception as e:
                logger.error(f"LLM初始化失败: {e}")
                self.llm = None
        else:
            self.llm = llm
            logger.info("使用用户提供的LLM")
            
        # 初始化图片索引映射表
        self.image_description_map = {}  # 图片路径 -> 描述文本
        self.description_image_map = {}  # 描述文本 -> 图片路径
        
        # 初始化处理统计信息
        self.processing_stats = {
            'total_pages_processed': 0,
            'total_images_extracted': 0,
            'total_images_merged': 0,
            'total_text_blocks_extracted': 0,
            'total_processing_time': 0.0,
            'average_time_per_page': 0.0,
            'description_generation_failures': 0,
            'image_extraction_failures': 0
        }
        
        logger.info("MultimodalPDFReader初始化完成")
        print(f"✅ PDF阅读器初始化成功")
        print(f"   📁 图片保存目录: {self.image_save_dir}")
        print(f"   🤖 LLM状态: {'已配置' if self.llm else '未配置'}")

    def _load_data(self, file: Path, extra_info: Optional[Dict] = None) -> List[DocNode]:
        """
        从PDF文件加载数据并提取文本和图片内容
        
        这是核心处理方法，负责：
        1. 打开和解析PDF文档
        2. 逐页提取文本块和图片
        3. 生成图片描述
        4. 创建DocNode对象
        5. 更新统计信息
        
        Args:
            file (Path): PDF文件路径
            extra_info (Optional[Dict]): 额外的元数据信息
            
        Returns:
            List[DocNode]: 包含文本和图片描述的DocNode对象列表
            
        Raises:
            FileNotFoundError: 当PDF文件不存在时
            fitz.FileDataError: 当PDF文件损坏或无法读取时
            Exception: 其他处理过程中的异常
        """
        logger.info(f"开始处理PDF文件: {file}")
        processing_start_time = time.time()
        
        # 确保文件路径是Path对象
        if not isinstance(file, Path):
            file = Path(file)
        
        # 检查文件是否存在
        if not file.exists():
            error_msg = f"PDF文件不存在: {file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 获取文件信息
        file_size = file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"PDF文件信息 - 大小: {file_size_mb:.2f} MB")
        print(f"📄 开始处理PDF: {file.name} ({file_size_mb:.2f} MB)")
        
        # 确保图片保存目录存在
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        try:
            # 打开PDF文档
            doc = fitz.open(file)
            logger.info(f"PDF文档打开成功 - 总页数: {len(doc)}")
            print(f"   📖 总页数: {len(doc)}")
            
        except fitz.FileDataError as e:
            error_msg = f"PDF文件损坏或无法读取: {e}"
            logger.error(error_msg)
            raise fitz.FileDataError(error_msg)
        except Exception as e:
            error_msg = f"打开PDF文件时发生未知错误: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        nodes = []
        image_count = 0
        
        try:
            # 逐页处理PDF内容
            for page_index in range(len(doc)):
                page_start_time = time.time()
                logger.info(f"处理第 {page_index + 1} 页...")
                
                try:
                    page = doc.load_page(page_index)
                    
                    # 提取文本块
                    logger.debug(f"第 {page_index + 1} 页：开始提取文本块")
                    text_blocks = self.extract_text_blocks(page)
                    text_block_count = len([b for b in text_blocks if b['content'].strip()])
                    self.processing_stats['total_text_blocks_extracted'] += text_block_count
                    logger.debug(f"第 {page_index + 1} 页：提取到 {text_block_count} 个文本块")
                    
                    # 提取和处理图片
                    logger.debug(f"第 {page_index + 1} 页：开始提取图片")
                    page_image_count = self._process_page_images(
                        page, page_index, text_blocks, nodes, image_count, extra_info, file
                    )
                    image_count += page_image_count
                    
                    # 添加文本节点
                    text_nodes_added = self._add_text_nodes(
                        text_blocks, page_index, nodes, extra_info, file
                    )
                    
                    page_time = time.time() - page_start_time
                    logger.info(f"第 {page_index + 1} 页处理完成 - 耗时: {page_time:.2f}秒, "
                              f"文本块: {text_nodes_added}, 图片: {page_image_count}")
                    
                    # 更新统计信息
                    self.processing_stats['total_pages_processed'] += 1
                    
                    # 显示进度
                    if (page_index + 1) % 5 == 0 or page_index == len(doc) - 1:
                        progress = ((page_index + 1) / len(doc)) * 100
                        print(f"   📊 处理进度: {progress:.1f}% ({page_index + 1}/{len(doc)})")
                    
                except Exception as e:
                    logger.error(f"处理第 {page_index + 1} 页时发生错误: {e}")
                    print(f"   ❌ 第 {page_index + 1} 页处理失败: {e}")
                    # 继续处理下一页，不中断整个流程
                    continue
            
        finally:
            # 确保文档被正确关闭
            doc.close()
            logger.info("PDF文档已关闭")
        
        # 计算总处理时间和统计信息
        total_time = time.time() - processing_start_time
        self.processing_stats['total_processing_time'] += total_time
        self.processing_stats['total_images_extracted'] = image_count
        
        if self.processing_stats['total_pages_processed'] > 0:
            self.processing_stats['average_time_per_page'] = (
                self.processing_stats['total_processing_time'] / 
                self.processing_stats['total_pages_processed']
            )
        
        # 输出处理总结
        logger.info(f"PDF处理完成 - 总耗时: {total_time:.2f}秒")
        logger.info(f"处理统计 - 页数: {len(nodes)}, 图片: {image_count}, "
                   f"文本块: {self.processing_stats['total_text_blocks_extracted']}")
        
        print(f"✅ PDF处理完成")
        print(f"   ⏱️  总耗时: {total_time:.2f}秒")
        print(f"   📄 处理页数: {self.processing_stats['total_pages_processed']}")
        print(f"   🖼️  提取图片: {image_count}张")
        print(f"   📝 文本块: {self.processing_stats['total_text_blocks_extracted']}个")
        print(f"   📊 平均每页耗时: {self.processing_stats['average_time_per_page']:.2f}秒")
        
        return nodes
    
    def _process_page_images(self, page, page_index: int, text_blocks: List[Dict], 
                           nodes: List[DocNode], current_image_count: int, 
                           extra_info: Optional[Dict], file: Path) -> int:
        """
        处理单页的图片内容
        
        Args:
            page: PyMuPDF页面对象
            page_index (int): 页面索引
            text_blocks (List[Dict]): 文本块列表
            nodes (List[DocNode]): 节点列表（用于添加新节点）
            current_image_count (int): 当前图片计数
            extra_info (Optional[Dict]): 额外元数据
            file (Path): PDF文件路径
            
        Returns:
            int: 本页处理的图片数量
        """
        logger.debug(f"开始处理第 {page_index + 1} 页的图片...")
        page_image_count = 0
        
        try:
            # 设置高分辨率矩阵（300 DPI）
            mat = fitz.Matrix(300/72, 300/72)
            
            # 获取页面图片信息
            img_infos = page.get_image_info(xrefs=True)
            logger.debug(f"第 {page_index + 1} 页发现 {len(img_infos)} 个图片对象")
            
            if not img_infos:
                return 0
            
            # 提取图片信息
            images = []
            for img_idx, img in enumerate(img_infos):
                try:
                    rect = fitz.Rect(img['bbox'])
                    if rect.is_empty:
                        logger.debug(f"跳过空矩形图片 {img_idx + 1}")
                        continue
                    
                    # 提取图片像素数据
                    pix = page.get_pixmap(matrix=mat, clip=rect)
                    
                    # 查找相邻文本
                    adjacent_texts = self.find_adjacent_text(rect, text_blocks)
                    context_before, context_after = self.extract_context_window(adjacent_texts)
                    
                    images.append({
                        'rect': rect,
                        'pix': pix,
                        'context_before': context_before,
                        'context_after': context_after,
                        'index': img_idx
                    })
                    
                except Exception as e:
                    logger.error(f"提取第 {img_idx + 1} 个图片时发生错误: {e}")
                    self.processing_stats['image_extraction_failures'] += 1
                    continue
            
            if not images:
                logger.debug(f"第 {page_index + 1} 页没有有效图片")
                return 0
            
            # 按垂直位置排序图片（从上到下）
            images.sort(key=lambda x: x['rect'].y0)
            logger.debug(f"第 {page_index + 1} 页有效图片数量: {len(images)}")
            
            # 图片分组和合并
            groups = self._group_mergeable_images(images)
            logger.debug(f"第 {page_index + 1} 页图片分组数量: {len(groups)}")
            
            # 处理每个图片组
            for group_idx, group in enumerate(groups):
                try:
                    if len(group) == 1:
                        # 单个图片处理
                        page_image_count += self._process_single_image(
                            group[0], page_index, current_image_count + page_image_count,
                            nodes, extra_info, file
                        )
                    else:
                        # 合并图片处理
                        page_image_count += self._process_merged_images(
                            group, page_index, current_image_count + page_image_count,
                            nodes, extra_info, file
                        )
                        self.processing_stats['total_images_merged'] += 1
                        
                except Exception as e:
                    logger.error(f"处理第 {group_idx + 1} 个图片组时发生错误: {e}")
                    continue
            
            logger.debug(f"第 {page_index + 1} 页图片处理完成，共处理 {page_image_count} 张图片")
            
        except Exception as e:
            logger.error(f"处理第 {page_index + 1} 页图片时发生未知错误: {e}")
            self.processing_stats['image_extraction_failures'] += 1
        
        return page_image_count
    
    def _group_mergeable_images(self, images: List[Dict]) -> List[List[Dict]]:
        """
        将可合并的图片分组
        
        Args:
            images (List[Dict]): 图片信息列表
            
        Returns:
            List[List[Dict]]: 图片分组列表
        """
        groups = []
        current_group = []
        threshold = 10  # 像素阈值
        
        for img in images:
            if not current_group:
                current_group.append(img)
            else:
                last = current_group[-1]
                # 检查垂直重叠和水平对齐
                vertical_overlap = (img['rect'].y0 < last['rect'].y1 + threshold)
                horizontal_overlap = (
                    max(img['rect'].x0, last['rect'].x0) < 
                    min(img['rect'].x1, last['rect'].x1)
                )
                
                if vertical_overlap and horizontal_overlap:
                    current_group.append(img)
                    logger.debug(f"图片 {img['index']} 与前一图片合并")
                else:
                    groups.append(current_group)
                    current_group = [img]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _process_single_image(self, img: Dict, page_index: int, image_count: int,
                            nodes: List[DocNode], extra_info: Optional[Dict], 
                            file: Path) -> int:
        """
        处理单个图片
        
        Args:
            img (Dict): 图片信息
            page_index (int): 页面索引
            image_count (int): 图片计数
            nodes (List[DocNode]): 节点列表
            extra_info (Optional[Dict]): 额外元数据
            file (Path): PDF文件路径
            
        Returns:
            int: 处理的图片数量（1或0）
        """
        try:
            # 生成图片文件名和路径
            img_filename = f'page{page_index+1}_img{image_count+1}.png'
            img_path = os.path.join(self.image_save_dir, img_filename)
            
            # 保存图片
            img['pix'].save(img_path)
            logger.debug(f"图片已保存: {img_path}")
            
            # 生成上下文和描述
            context_before = img['context_before']
            context_after = img['context_after']
            context = f"{context_before} [IMAGE] {context_after}" if context_before or context_after else "[IMAGE]"
            
            # 生成图片描述
            description = self._generate_image_description(context, "image")
            
            # 处理路径信息
            img_path_abs = os.path.abspath(img_path)
            img_filename = os.path.basename(img_path)
            
            # 更新图片索引映射表
            self.image_description_map[img_path_abs] = description
            self.description_image_map[description] = img_path_abs
            
            # 创建元数据
            metadata = (extra_info or {}).copy()
            metadata.update({
                'file_name': file.name,
                'page': page_index + 1,
                'bbox': list(img['rect']),
                'image_path': img_path_abs,
                'image_filename': img_filename,
                'image_relative_path': os.path.relpath(img_path_abs, self.image_save_dir),
                'type': 'image_description',
                'original_context_before': context_before,
                'original_context_after': context_after,
                'image_size': f"{img['pix'].width}x{img['pix'].height}",
                'processing_timestamp': time.time()
            })
            
            # 创建并添加节点
            nodes.append(DocNode(text=description, metadata=metadata))
            logger.debug(f"单个图片节点已创建: {img_filename}")
            
            return 1
            
        except Exception as e:
            logger.error(f"处理单个图片时发生错误: {e}")
            self.processing_stats['image_extraction_failures'] += 1
            return 0
    
    def _process_merged_images(self, group: List[Dict], page_index: int, image_count: int,
                             nodes: List[DocNode], extra_info: Optional[Dict], 
                             file: Path) -> int:
        """
        处理合并图片组
        
        Args:
            group (List[Dict]): 图片组
            page_index (int): 页面索引
            image_count (int): 图片计数
            nodes (List[DocNode]): 节点列表
            extra_info (Optional[Dict]): 额外元数据
            file (Path): PDF文件路径
            
        Returns:
            int: 处理的图片数量（1或0）
        """
        try:
            # 按垂直位置排序
            group.sort(key=lambda x: x['rect'].y0)
            
            # 计算合并后的尺寸
            merged_width = max(int(g['rect'].width * (300/72)) for g in group)
            merged_height = sum(int(g['rect'].height * (300/72)) for g in group)
            
            logger.debug(f"合并图片尺寸: {merged_width}x{merged_height}")
            
            # 创建合并图片
            merged_img = Image.new('RGB', (merged_width, merged_height), color='white')
            y_offset = 0
            
            for g in group:
                try:
                    # 转换为PIL图片
                    pil_img = Image.frombytes('RGB', (g['pix'].width, g['pix'].height), g['pix'].samples)
                    
                    # 调整宽度以匹配合并图片
                    if pil_img.width != merged_width:
                        pil_img = pil_img.resize((merged_width, pil_img.height), Image.Resampling.LANCZOS)
                    
                    # 粘贴到合并图片
                    merged_img.paste(pil_img, (0, y_offset))
                    y_offset += pil_img.height
                    
                except Exception as e:
                    logger.error(f"合并图片片段时发生错误: {e}")
                    continue
            
            # 生成文件名和路径
            img_filename = f'page{page_index+1}_merged{image_count+1}.png'
            img_path = os.path.join(self.image_save_dir, img_filename)
            
            # 保存合并图片
            merged_img.save(img_path, 'PNG', quality=95)
            logger.debug(f"合并图片已保存: {img_path}")
            
            # 合并上下文信息
            context_before = ' '.join(g['context_before'] for g in group if g['context_before'])
            context_after = ' '.join(g['context_after'] for g in group if g['context_after'])
            context = f"{context_before} [MERGED_IMAGE] {context_after}" if context_before or context_after else "[MERGED_IMAGE]"
            
            # 生成图片描述
            description = self._generate_image_description(context, "merged_image")
            
            # 处理路径信息
            img_path_abs = os.path.abspath(img_path)
            
            # 更新图片索引映射表
            self.image_description_map[img_path_abs] = description
            self.description_image_map[description] = img_path_abs
            
            # 计算合并区域的边界框
            combined_rect = fitz.Rect(
                min(g['rect'].x0 for g in group),
                group[0]['rect'].y0,
                max(g['rect'].x1 for g in group),
                group[-1]['rect'].y1
            )
            
            # 创建元数据
            metadata = (extra_info or {}).copy()
            metadata.update({
                'file_name': file.name,
                'page': page_index + 1,
                'bbox': list(combined_rect),
                'image_path': img_path_abs,
                'image_filename': img_filename,
                'image_relative_path': os.path.relpath(img_path_abs, self.image_save_dir),
                'type': 'image_description',
                'original_context_before': context_before,
                'original_context_after': context_after,
                'image_size': f"{merged_width}x{merged_height}",
                'merged_count': len(group),
                'processing_timestamp': time.time()
            })
            
            # 创建并添加节点
            nodes.append(DocNode(text=description, metadata=metadata))
            logger.debug(f"合并图片节点已创建: {img_filename} (合并了{len(group)}张图片)")
            
            return 1
            
        except Exception as e:
            logger.error(f"处理合并图片时发生错误: {e}")
            self.processing_stats['image_extraction_failures'] += 1
            return 0
    
    def _generate_image_description(self, context: str, image_type: str) -> str:
        """
        生成图片描述
        
        Args:
            context (str): 图片上下文
            image_type (str): 图片类型（"image" 或 "merged_image"）
            
        Returns:
            str: 图片描述文本
        """
        try:
            if self.llm is None:
                logger.warning("LLM未配置，使用默认描述")
                return f"图片描述生成器未配置，上下文：{context}"
            
            # 构建提示词
            if image_type == "merged_image":
                prompt = f"Based on the following context, generate a concise description of the merged image: {context}"
            else:
                prompt = f"Based on the following context, generate a concise description of the image: {context}"
            
            logger.debug(f"生成图片描述，提示词长度: {len(prompt)}")
            
            # 调用LLM生成描述
            description = self.llm(prompt)
            
            # 验证描述质量
            if not description or not description.strip():
                logger.warning("LLM返回空描述")
                description = f"图片描述生成失败，上下文：{context}"
            else:
                logger.debug(f"图片描述生成成功，长度: {len(description)}")
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"图片描述生成时发生错误: {e}")
            self.processing_stats['description_generation_failures'] += 1
            
            # 返回包含错误信息的描述
            error_description = f"图片处理错误：{str(e)}，上下文：{context}"
            return error_description
    
    def _add_text_nodes(self, text_blocks: List[Dict], page_index: int, 
                       nodes: List[DocNode], extra_info: Optional[Dict], 
                       file: Path) -> int:
        """
        添加文本节点
        
        Args:
            text_blocks (List[Dict]): 文本块列表
            page_index (int): 页面索引
            nodes (List[DocNode]): 节点列表
            extra_info (Optional[Dict]): 额外元数据
            file (Path): PDF文件路径
            
        Returns:
            int: 添加的文本节点数量
        """
        text_nodes_added = 0
        
        for block in text_blocks:
            if block['content'].strip():
                try:
                    # 创建元数据
                    metadata = (extra_info or {}).copy()
                    metadata.update({
                        'file_name': file.name,
                        'page': page_index + 1,
                        'bbox': block['bbox'],
                        'type': 'text',
                        'text_length': len(block['content']),
                        'processing_timestamp': time.time()
                    })
                    
                    # 创建并添加文本节点
                    nodes.append(DocNode(text=block['content'], metadata=metadata))
                    text_nodes_added += 1
                    
                except Exception as e:
                    logger.error(f"创建文本节点时发生错误: {e}")
                    continue
        
        logger.debug(f"第 {page_index + 1} 页添加了 {text_nodes_added} 个文本节点")
        return text_nodes_added

    def extract_text_blocks(self, page) -> List[Dict]:
        """
        从页面提取文本块
        
        使用PyMuPDF的字典模式提取文本，保留位置和格式信息
        
        Args:
            page: PyMuPDF页面对象
            
        Returns:
            List[Dict]: 文本块信息列表，每个字典包含：
                - type: 'text'
                - bbox: 边界框坐标 [x0, y0, x1, y1]
                - content: 文本内容
                - x0, y0, x1, y1: 边界框坐标（便于访问）
        """
        logger.debug("开始提取文本块...")
        text_blocks = []
        
        try:
            # 获取页面文本字典
            blocks = page.get_text('dict')['blocks']
            
            for block_idx, block in enumerate(blocks):
                try:
                    # 只处理包含文本行的块
                    if 'lines' in block:
                        bbox = block['bbox']
                        text_content = ''
                        
                        # 遍历文本行和跨度
                        for line in block['lines']:
                            for span in line['spans']:
                                text_content += span['text']
                            text_content += '\n'
                        
                        # 清理文本内容
                        text_content = text_content.strip()
                        
                        if text_content:  # 只添加非空文本块
                            text_blocks.append({
                                'type': 'text',
                                'bbox': bbox,
                                'content': text_content,
                                'x0': bbox[0], 'y0': bbox[1], 
                                'x1': bbox[2], 'y1': bbox[3],
                                'block_index': block_idx
                            })
                            
                except Exception as e:
                    logger.error(f"处理文本块 {block_idx} 时发生错误: {e}")
                    continue
            
            logger.debug(f"提取到 {len(text_blocks)} 个有效文本块")
            
        except Exception as e:
            logger.error(f"提取文本块时发生错误: {e}")
        
        return text_blocks
    
    def calculate_distance(self, bbox1: Tuple[float, float, float, float], 
                          bbox2: Tuple[float, float, float, float]) -> float:
        """
        计算两个边界框之间的距离
        
        使用欧几里得距离计算两个矩形边界框之间的最短距离
        
        Args:
            bbox1: 第一个边界框 (x0, y0, x1, y1)
            bbox2: 第二个边界框 (x0, y0, x1, y1)
            
        Returns:
            float: 两个边界框之间的距离（像素）
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算水平和垂直方向的距离
        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        
        # 返回欧几里得距离
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance
    
    def find_adjacent_text(self, img_bbox: Tuple[float, float, float, float], 
                          text_blocks: List[Dict], max_distance: float = 200) -> List[Dict]:
        """
        查找图片附近的文本块
        
        Args:
            img_bbox: 图片边界框
            text_blocks: 文本块列表
            max_distance: 最大搜索距离（像素）
            
        Returns:
            List[Dict]: 相邻文本信息列表，按距离排序
        """
        logger.debug(f"查找图片附近的文本，最大距离: {max_distance}")
        adjacent_texts = []
        
        for text_block in text_blocks:
            # 跳过空文本块
            if not text_block['content'].strip():
                continue
            
            try:
                # 计算距离
                distance = self.calculate_distance(img_bbox, text_block['bbox'])
                
                if distance <= max_distance:
                    # 确定文本相对于图片的位置
                    img_center_y = (img_bbox[1] + img_bbox[3]) / 2
                    text_center_y = (text_block['y0'] + text_block['y1']) / 2
                    position = 'below' if text_center_y > img_center_y else 'above'
                    
                    adjacent_texts.append({
                        'text_block': text_block,
                        'distance': distance,
                        'position': position
                    })
                    
            except Exception as e:
                logger.error(f"计算文本块距离时发生错误: {e}")
                continue
        
        # 按距离排序，取最近的20个
        adjacent_texts.sort(key=lambda x: x['distance'])
        result = adjacent_texts[:20]
        
        logger.debug(f"找到 {len(result)} 个相邻文本块")
        return result
    
    def extract_context_window(self, adjacent_texts: List[Dict], 
                             context_window_before: int = 100, 
                             context_window_after: int = 100) -> Tuple[str, str]:
        """
        提取上下文窗口
        
        从相邻文本中提取图片前后的上下文信息
        
        Args:
            adjacent_texts: 相邻文本信息列表
            context_window_before: 图片前上下文的最大字符数
            context_window_after: 图片后上下文的最大字符数
            
        Returns:
            Tuple[str, str]: (前置上下文, 后置上下文)
        """
        logger.debug(f"提取上下文窗口，前置: {context_window_before}, 后置: {context_window_after}")
        
        context_before = ''
        context_after = ''
        
        try:
            # 分离图片上方和下方的文本
            texts_above = [t for t in adjacent_texts if t['position'] == 'above']
            texts_below = [t for t in adjacent_texts if t['position'] == 'below']
            
            # 提取图片前的上下文（图片上方的文本）
            if texts_above:
                combined = ' '.join(t['text_block']['content'] for t in texts_above)
                if len(combined) > context_window_before:
                    context_before = combined[-context_window_before:]
                else:
                    context_before = combined
            
            # 提取图片后的上下文（图片下方的文本）
            if texts_below:
                combined = ' '.join(t['text_block']['content'] for t in texts_below)
                if len(combined) > context_window_after:
                    context_after = combined[:context_window_after]
                else:
                    context_after = combined
            
            logger.debug(f"上下文提取完成，前置长度: {len(context_before)}, 后置长度: {len(context_after)}")
            
        except Exception as e:
            logger.error(f"提取上下文窗口时发生错误: {e}")
        
        return context_before.strip(), context_after.strip()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict[str, Any]: 包含各种处理统计信息的字典
        """
        return self.processing_stats.copy()
    
    def get_image_mappings(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        获取图片映射表
        
        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: (图片路径->描述映射, 描述->图片路径映射)
        """
        return self.image_description_map.copy(), self.description_image_map.copy()
    
    def clear_cache(self):
        """清理缓存和临时数据"""
        logger.info("清理MultimodalPDFReader缓存...")
        self.image_description_map.clear()
        self.description_image_map.clear()
        
        # 重置统计信息
        self.processing_stats = {
            'total_pages_processed': 0,
            'total_images_extracted': 0,
            'total_images_merged': 0,
            'total_text_blocks_extracted': 0,
            'total_processing_time': 0.0,
            'average_time_per_page': 0.0,
            'description_generation_failures': 0,
            'image_extraction_failures': 0
        }
        
        logger.info("缓存清理完成")
        print("🧹 PDF阅读器缓存已清理")


# 模块级别的便利函数
def create_multimodal_pdf_reader(image_save_dir: str = './images', 
                                llm: Any = None) -> MultimodalPDFReader:
    """
    创建多模态PDF阅读器的便利函数
    
    Args:
        image_save_dir (str): 图片保存目录
        llm (Any): 语言模型实例
        
    Returns:
        MultimodalPDFReader: 配置好的PDF阅读器实例
    """
    logger.info(f"创建多模态PDF阅读器 - 图片目录: {image_save_dir}")
    return MultimodalPDFReader(image_save_dir=image_save_dir, llm=llm)


# 模块初始化日志
logger.info("MultimodalPDFReader模块加载完成")
print("📚 多模态PDF阅读器模块已加载")