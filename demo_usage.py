#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG Pipeline使用示例和测试脚本

本脚本演示了如何使用MultimodalRAGPipeline进行PDF文档的多模态问答：
1. 基本的Pipeline创建和初始化
2. 知识库构建和文档索引
3. 多轮问答交互
4. 图片检索和显示
5. 对话历史管理
6. 错误处理和性能监控

功能特性：
- 支持多种查询类型（文本、图片相关）
- 自动图片检索和描述生成
- 完整的对话历史记录
- 详细的调试信息输出
- 性能指标统计

作者: AI Assistant
版本: 2.0
更新时间: 2025-01-22
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional
import traceback

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demo_usage.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入自定义模块
try:
    from newpip import MultimodalRAGPipeline, create_multimodal_pipeline
    logger.info("成功导入MultimodalRAGPipeline相关模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    print(f"错误：无法导入必要的模块 - {e}")
    sys.exit(1)

class DemoRunner:
    """
    演示运行器类
    
    负责管理和执行多模态RAG Pipeline的演示流程，包括：
    1. 环境检查和初始化
    2. Pipeline创建和配置
    3. 测试用例执行
    4. 性能监控和统计
    5. 结果展示和分析
    
    Attributes:
        pdf_path (str): PDF文件路径
        image_save_dir (str): 图片保存目录
        pipeline (MultimodalRAGPipeline): Pipeline实例
        performance_stats (Dict): 性能统计数据
        test_queries (List[str]): 测试查询列表
    """
    
    def __init__(self, pdf_path: str = None, image_save_dir: str = None):
        """
        初始化演示运行器
        
        Args:
            pdf_path (str, optional): PDF文件路径，默认为"./test.pdf"
            image_save_dir (str, optional): 图片保存目录，默认为"./images"
        """
        logger.info("初始化DemoRunner...")
        
        # 设置默认路径
        self.pdf_path = pdf_path or os.path.abspath("./test.pdf")
        self.image_save_dir = image_save_dir or os.path.abspath("./images")
        
        logger.info(f"PDF路径: {self.pdf_path}")
        logger.info(f"图片目录: {self.image_save_dir}")
        
        # 初始化变量
        self.pipeline = None
        self.performance_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_time': 0.0,
            'average_response_time': 0.0,
            'images_retrieved': 0,
            'knowledge_base_build_time': 0.0
        }
        
        # 预定义测试查询
        self.test_queries = [
            "这个文档的主要内容是什么？",
            "文档中有哪些图片？请描述一下。",
            "能否详细解释一下相关的技术原理？",
            "文档中提到了哪些重要概念？",
            "有没有相关的数据或统计信息？"
        ]
        
        logger.info("DemoRunner初始化完成")
    
    def check_environment(self) -> bool:
        """
        检查运行环境
        
        验证必要的文件和目录是否存在，以及系统配置是否正确
        
        Returns:
            bool: 环境检查是否通过
        """
        logger.info("开始环境检查...")
        
        # 检查PDF文件
        if not os.path.exists(self.pdf_path):
            logger.error(f"PDF文件不存在: {self.pdf_path}")
            print(f"❌ 错误：PDF文件不存在 - {self.pdf_path}")
            return False
        
        logger.info(f"✅ PDF文件存在: {self.pdf_path}")
        print(f"✅ PDF文件检查通过: {os.path.basename(self.pdf_path)}")
        
        # 检查并创建图片目录
        try:
            os.makedirs(self.image_save_dir, exist_ok=True)
            logger.info(f"✅ 图片目录已创建/确认: {self.image_save_dir}")
            print(f"✅ 图片目录检查通过: {self.image_save_dir}")
        except OSError as e:
            logger.error(f"创建图片目录失败: {e}")
            print(f"❌ 错误：无法创建图片目录 - {e}")
            return False
        
        # 检查文件大小
        try:
            file_size = os.path.getsize(self.pdf_path)
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"PDF文件大小: {file_size_mb:.2f} MB")
            print(f"📄 PDF文件大小: {file_size_mb:.2f} MB")
            
            if file_size_mb > 100:
                logger.warning("PDF文件较大，处理可能需要更长时间")
                print("⚠️  警告：PDF文件较大，处理可能需要更长时间")
        except OSError as e:
            logger.error(f"获取文件大小失败: {e}")
        
        logger.info("环境检查完成")
        return True
    
    def create_pipeline(self) -> bool:
        """
        创建Pipeline实例
        
        Returns:
            bool: Pipeline创建是否成功
        """
        logger.info("开始创建Pipeline实例...")
        print("\n🔧 正在创建多模态RAG Pipeline...")
        
        try:
            start_time = time.time()
            
            # 创建Pipeline实例
            self.pipeline = MultimodalRAGPipeline(self.pdf_path, self.image_save_dir)
            
            creation_time = time.time() - start_time
            logger.info(f"Pipeline创建成功 - 耗时: {creation_time:.2f}秒")
            print(f"✅ Pipeline创建成功 (耗时: {creation_time:.2f}秒)")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline创建失败: {e}")
            print(f"❌ Pipeline创建失败: {e}")
            traceback.print_exc()
            return False
    
    def build_knowledge_base(self) -> bool:
        """
        构建知识库
        
        Returns:
            bool: 知识库构建是否成功
        """
        logger.info("开始构建知识库...")
        print("\n📚 正在构建知识库...")
        
        if self.pipeline is None:
            logger.error("Pipeline未初始化")
            print("❌ 错误：Pipeline未初始化")
            return False
        
        try:
            start_time = time.time()
            
            # 构建知识库
            self.pipeline.build_knowledge_base()
            
            build_time = time.time() - start_time
            self.performance_stats['knowledge_base_build_time'] = build_time
            
            # 获取统计信息
            image_count = len(self.pipeline.image_description_map)
            
            logger.info(f"知识库构建成功 - 耗时: {build_time:.2f}秒, 图片数量: {image_count}")
            print(f"✅ 知识库构建成功")
            print(f"   📊 构建耗时: {build_time:.2f}秒")
            print(f"   🖼️  图片数量: {image_count}张")
            
            return True
            
        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            print(f"❌ 知识库构建失败: {e}")
            traceback.print_exc()
            return False
    
    def execute_query(self, query: str, query_index: int = 0) -> Optional[Dict[str, Any]]:
        """
        执行单个查询
        
        Args:
            query (str): 查询文本
            query_index (int): 查询索引（用于显示）
            
        Returns:
            Optional[Dict[str, Any]]: 查询结果，失败时返回None
        """
        logger.info(f"执行查询 {query_index + 1}: {query}")
        print(f"\n❓ 查询 {query_index + 1}: {query}")
        
        if self.pipeline is None:
            logger.error("Pipeline未初始化")
            print("❌ 错误：Pipeline未初始化")
            return None
        
        try:
            start_time = time.time()
            
            # 执行查询
            result = self.pipeline.query_and_response(query)
            
            query_time = time.time() - start_time
            
            # 更新统计信息
            self.performance_stats['total_queries'] += 1
            self.performance_stats['successful_queries'] += 1
            self.performance_stats['total_time'] += query_time
            self.performance_stats['images_retrieved'] += len(result.get('images', []))
            
            # 计算平均响应时间
            if self.performance_stats['successful_queries'] > 0:
                self.performance_stats['average_response_time'] = (
                    self.performance_stats['total_time'] / self.performance_stats['successful_queries']
                )
            
            logger.info(f"查询成功 - 耗时: {query_time:.2f}秒, 图片数量: {len(result.get('images', []))}")
            
            # 显示响应
            print(f"💬 回答: {result['response']}")
            print(f"⏱️  响应时间: {query_time:.2f}秒")
            
            # 显示相关图片
            images = result.get('images', [])
            if images:
                print(f"\n🖼️  相关图片 ({len(images)}张):")
                
                # 准备图片显示信息
                display_images = self.pipeline.render_images_for_display(images)
                
                for j, img in enumerate(display_images, 1):
                    print(f"   图片 {j}:")
                    print(f"     📁 文件名: {img['filename']}")
                    print(f"     📍 路径: {img['display_url']}")
                    print(f"     📏 大小: {img['size']}")
                    print(f"     📝 描述: {img['description'][:100]}{'...' if len(img['description']) > 100 else ''}")
            else:
                print("   🔍 没有找到相关图片")
            
            return result
            
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            print(f"❌ 查询执行失败: {e}")
            
            # 更新失败统计
            self.performance_stats['total_queries'] += 1
            self.performance_stats['failed_queries'] += 1
            
            traceback.print_exc()
            return None
    
    def run_test_queries(self) -> bool:
        """
        运行预定义的测试查询
        
        Returns:
            bool: 测试是否成功完成
        """
        logger.info("开始运行测试查询...")
        print(f"\n🧪 开始执行 {len(self.test_queries)} 个测试查询...")
        
        success_count = 0
        
        for i, query in enumerate(self.test_queries):
            print(f"\n{'='*60}")
            
            result = self.execute_query(query, i)
            if result is not None:
                success_count += 1
            
            # 添加分隔线
            print(f"{'='*60}")
            
            # 短暂暂停，避免过快执行
            time.sleep(0.5)
        
        # 显示测试总结
        print(f"\n📊 测试总结:")
        print(f"   ✅ 成功查询: {success_count}/{len(self.test_queries)}")
        print(f"   ❌ 失败查询: {len(self.test_queries) - success_count}/{len(self.test_queries)}")
        
        logger.info(f"测试查询完成 - 成功: {success_count}, 失败: {len(self.test_queries) - success_count}")
        
        return success_count == len(self.test_queries)
    
    def show_conversation_history(self):
        """显示对话历史"""
        logger.info("显示对话历史...")
        print(f"\n📜 对话历史记录:")
        
        if self.pipeline is None:
            print("   ❌ Pipeline未初始化")
            return
        
        history = self.pipeline.get_conversation_history()
        
        if not history:
            print("   📭 暂无对话记录")
            return
        
        print(f"   📈 总对话数: {len(history)}")
        print(f"   {'='*50}")
        
        for i, conv in enumerate(history, 1):
            timestamp = conv.get('timestamp', '未知时间')
            print(f"   对话 {i} ({timestamp}):")
            print(f"     ❓ 问题: {conv['query']}")
            print(f"     💬 回答: {conv['response'][:150]}{'...' if len(conv['response']) > 150 else ''}")
            print(f"     🖼️  图片数: {len(conv.get('images', []))}")
            print(f"     📏 上下文长度: {len(conv.get('context', ''))} 字符")
            print()
    
    def show_performance_stats(self):
        """显示性能统计"""
        logger.info("显示性能统计...")
        print(f"\n📈 性能统计报告:")
        print(f"   {'='*50}")
        
        stats = self.performance_stats
        
        print(f"   🔢 查询统计:")
        print(f"     总查询数: {stats['total_queries']}")
        print(f"     成功查询: {stats['successful_queries']}")
        print(f"     失败查询: {stats['failed_queries']}")
        
        if stats['successful_queries'] > 0:
            success_rate = (stats['successful_queries'] / stats['total_queries']) * 100
            print(f"     成功率: {success_rate:.1f}%")
        
        print(f"\n   ⏱️  时间统计:")
        print(f"     知识库构建时间: {stats['knowledge_base_build_time']:.2f}秒")
        print(f"     总查询时间: {stats['total_time']:.2f}秒")
        print(f"     平均响应时间: {stats['average_response_time']:.2f}秒")
        
        print(f"\n   🖼️  图片统计:")
        print(f"     检索到的图片总数: {stats['images_retrieved']}")
        
        if stats['successful_queries'] > 0:
            avg_images = stats['images_retrieved'] / stats['successful_queries']
            print(f"     平均每次查询图片数: {avg_images:.1f}")
    
    def run_demo(self) -> bool:
        """
        运行完整的演示流程
        
        Returns:
            bool: 演示是否成功完成
        """
        logger.info("开始运行完整演示...")
        print("🚀 多模态RAG Pipeline 演示程序")
        print("=" * 60)
        
        # 1. 环境检查
        if not self.check_environment():
            logger.error("环境检查失败")
            return False
        
        # 2. 创建Pipeline
        if not self.create_pipeline():
            logger.error("Pipeline创建失败")
            return False
        
        # 3. 构建知识库
        if not self.build_knowledge_base():
            logger.error("知识库构建失败")
            return False
        
        # 4. 运行测试查询
        test_success = self.run_test_queries()
        
        # 5. 显示对话历史
        self.show_conversation_history()
        
        # 6. 显示性能统计
        self.show_performance_stats()
        
        # 7. 演示总结
        print(f"\n🎯 演示总结:")
        if test_success:
            print("   ✅ 所有测试查询执行成功")
            logger.info("演示成功完成")
        else:
            print("   ⚠️  部分测试查询执行失败")
            logger.warning("演示部分成功")
        
        print("   📝 详细日志已保存到 demo_usage.log")
        print("=" * 60)
        
        return test_success


def test_simple_interface():
    """
    测试简化接口（向后兼容性测试）
    """
    logger.info("开始测试简化接口...")
    print(f"\n🔧 测试简化接口 (向后兼容性)...")
    
    try:
        # 配置路径
        pdf_path = os.path.abspath("./test.pdf")
        image_save_dir = os.path.abspath("./images")
        
        if not os.path.exists(pdf_path):
            print(f"❌ 简化接口测试跳过：PDF文件不存在 - {pdf_path}")
            return False
        
        print("   📚 创建简化Pipeline...")
        start_time = time.time()
        
        # 使用简化接口
        simple_pipeline = create_multimodal_pipeline(pdf_path, image_save_dir)
        simple_pipeline.build_knowledge_base()
        
        setup_time = time.time() - start_time
        
        # 执行简单查询
        simple_query = "请总结一下文档的核心观点。"
        print(f"   ❓ 测试查询: {simple_query}")
        
        query_start = time.time()
        simple_result = simple_pipeline.query_and_response(simple_query)
        query_time = time.time() - query_start
        
        print(f"   💬 回答: {simple_result['response'][:200]}{'...' if len(simple_result['response']) > 200 else ''}")
        print(f"   ⏱️  设置时间: {setup_time:.2f}秒")
        print(f"   ⏱️  查询时间: {query_time:.2f}秒")
        print("   ✅ 简化接口测试成功")
        
        logger.info(f"简化接口测试成功 - 设置时间: {setup_time:.2f}秒, 查询时间: {query_time:.2f}秒")
        return True
        
    except Exception as e:
        logger.error(f"简化接口测试失败: {e}")
        print(f"   ❌ 简化接口测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """
    主函数：演示多模态RAG Pipeline的使用
    
    执行完整的演示流程，包括：
    1. 环境检查和初始化
    2. Pipeline创建和知识库构建
    3. 多轮问答测试
    4. 性能统计和结果分析
    5. 向后兼容性测试
    """
    logger.info("=" * 60)
    logger.info("启动多模态RAG Pipeline演示程序")
    logger.info("=" * 60)
    
    # 记录程序开始时间
    program_start_time = time.time()
    
    try:
        # 配置参数 - 使用绝对路径
        pdf_path = os.path.abspath("./test.pdf")
        image_save_dir = os.path.abspath("./images")
        
        logger.info(f"配置参数 - PDF路径: {pdf_path}, 图片目录: {image_save_dir}")
        
        # 创建并运行演示
        demo_runner = DemoRunner(pdf_path, image_save_dir)
        demo_success = demo_runner.run_demo()
        
        # 测试简化接口
        simple_success = test_simple_interface()
        
        # 计算总运行时间
        total_time = time.time() - program_start_time
        
        # 最终总结
        print(f"\n🏁 程序执行完成:")
        print(f"   ⏱️  总运行时间: {total_time:.2f}秒")
        print(f"   📊 主要演示: {'✅ 成功' if demo_success else '❌ 失败'}")
        print(f"   🔧 简化接口: {'✅ 成功' if simple_success else '❌ 失败'}")
        
        logger.info(f"程序执行完成 - 总时间: {total_time:.2f}秒, 主演示: {demo_success}, 简化接口: {simple_success}")
        
        # 返回适当的退出码
        if demo_success and simple_success:
            logger.info("所有测试成功完成")
            return 0
        else:
            logger.warning("部分测试失败")
            return 1
            
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        print("\n⚠️  程序被用户中断")
        return 130
        
    except Exception as e:
        logger.error(f"程序执行出现未预期错误: {e}")
        print(f"\n❌ 程序执行出现错误: {e}")
        traceback.print_exc()
        return 1
    
    finally:
        logger.info("程序清理完成")


if __name__ == "__main__":
    """程序入口点"""
    exit_code = main()
    sys.exit(exit_code)