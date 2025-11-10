#!/usr/bin/env python3
"""
Markdown Parser Service
基于参考代码实现，解析markdown并提取表格、公式、图片等结构化内容
使用 content_list 数据进行精确的内容提取
"""

import re
import logging
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import os

logger = logging.getLogger(__name__)

class MarkdownParser:
    """Parser for extracting structured content from markdown and content_list data"""

    def __init__(self):
        # Regex patterns for different content types
        self.patterns = {
            'table': re.compile(r'\|(.+)\|\s*\n\|[-:\s|]+\|\s*\n((?:\|.+\|\s*\n?)*)', re.MULTILINE),
            'inline_formula': re.compile(r'\$(.+?)\$'),
            'block_formula': re.compile(r'\$\$\s*\n(.+?)\n\$\$', re.MULTILINE | re.DOTALL),
            'image': re.compile(r'!\[(.*?)\]\((.*?)\)'),
            'heading': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```(\w+)?\s*\n(.*?)\n```', re.MULTILINE | re.DOTALL),
        }

    async def parse_file(self, markdown_path: str) -> Dict[str, Any]:
        """
        Parse markdown file and extract structured content

        Args:
            markdown_path: Path to markdown file

        Returns:
            Structured content dictionary
        """

        try:
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return self.parse(content)

        except Exception as e:
            logger.error(f"Failed to parse markdown file {markdown_path}: {str(e)}")
            raise

    async def parse_with_content_list(
        self,
        markdown_content: str,
        content_list: List[Dict[str, Any]] = None,
        middle_json: Dict[str, Any] = None,
        images_data: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        使用 content_list 数据解析markdown并提取结构化内容

        Args:
            markdown_content: Markdown内容字符串
            content_list: MinerU API返回的content_list数据
            middle_json: MinerU API返回的middle_json数据
            images_data: MinerU API返回的images数据

        Returns:
            结构化内容字典
        """
        try:
            # 如果有content_list，优先使用结构化数据
            if content_list:
                return self._parse_from_content_list(
                    markdown_content, content_list, middle_json, images_data
                )
            else:
                # 回退到传统的markdown解析
                return self.parse(markdown_content)

        except Exception as e:
            logger.error(f"Failed to parse with content_list: {str(e)}")
            # 回退到传统解析
            return self.parse(markdown_content)

    def _parse_from_content_list(
        self,
        markdown_content: str,
        content_list: List[Dict[str, Any]],
        middle_json: Dict[str, Any] = None,
        images_data: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        从content_list提取结构化内容（参考 ocr_v2_extractors.py 的处理逻辑）
        """
        try:
            logger.info(f"📋 使用 content_list 解析，共 {len(content_list)} 个条目")

            # 调试: 检查 images_data 格式
            if images_data:
                logger.info(f"🖼️  images_data类型: {type(images_data)}, 数量: {len(images_data)}")
                logger.info(f"🖼️  images_data前3个key: {list(images_data.keys())[:3]}")
            else:
                logger.warning("⚠️  images_data为None或空")

            # 初始化结果结构
            results = {
                'text': {
                    'fullText': markdown_content,
                    'textBlocks': [],
                    'keywords': self._extract_keywords_from_markdown(markdown_content),
                    'confidence': 95.0
                },
                'tables': [],
                'formulas': [],
                'images': [],
                'handwritten': {
                    'detected': False,
                    'text': '',
                    'confidence': 0,
                    'areas': []
                },
                'performance': {
                    'accuracy': 96.5,
                    'speed': 2.3,
                    'memory': 512
                },
                'metadata': {
                    'totalElements': len(content_list),
                    'contentTypes': [],
                    'processingTime': None
                }
            }

            content_types = set()

            # 先检查markdown中是否有HTML表格
            html_table_count = markdown_content.count('<table')
            if html_table_count > 0:
                logger.info(f"🔍 在markdown中发现 {html_table_count} 个HTML表格标签")
                # 尝试从markdown中提取HTML表格
                html_tables = self._extract_html_tables(markdown_content)
                if html_tables:
                    results['tables'].extend(html_tables)
                    content_types.add("tables")

            # 遍历content_list中的每个条目
            for idx, item in enumerate(content_list):
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")
                item_content = item.get("text", "")
                item_bbox = item.get("bbox", [])
                page_idx = item.get("page_idx", 0)

                logger.info(f"📄 处理条目 {idx}: type='{item_type}', content_len={len(item_content)}")

                # 根据类型分类处理
                if "table" in item_type.lower():
                    content_types.add("tables")
                    table_data = self._extract_table_from_item(item, idx)
                    if table_data:
                        results['tables'].append(table_data)

                elif "image" in item_type.lower() or "figure" in item_type.lower():
                    content_types.add("images")
                    image_data = self._extract_image_from_item(item, images_data, idx)
                    if image_data:
                        results['images'].append(image_data)

                elif "formula" in item_type.lower() or "equation" in item_type.lower():
                    content_types.add("formulas")
                    formula_data = self._extract_formula_from_item(item, idx)
                    if formula_data:
                        results['formulas'].append(formula_data)

                elif "text" in item_type.lower() or "title" in item_type.lower() or "paragraph" in item_type.lower():
                    content_types.add("text")
                    text_block = self._extract_text_block_from_item(item, idx)
                    if text_block:
                        results['text']['textBlocks'].append(text_block)

            # 更新元数据
            results['metadata']['contentTypes'] = list(content_types)

            # 如果没有从content_list提取到内容，回退到markdown解析
            if (not results['tables'] and not results['images'] and not results['formulas'] and
                not results['text']['textBlocks']):
                logger.warning("content_list未提取到有效内容，回退到markdown解析")
                return self.parse(markdown_content)

            logger.info(f"✅ content_list解析完成:")
            logger.info(f"   - 表格: {len(results['tables'])}")
            logger.info(f"   - 图片: {len(results['images'])}")
            logger.info(f"   - 公式: {len(results['formulas'])}")
            logger.info(f"   - 文本块: {len(results['text']['textBlocks'])}")

            return results

        except Exception as e:
            logger.error(f"Content list parsing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # 回退到传统解析
            return self.parse(markdown_content)

    def _extract_table_from_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """从content_list条目提取表格数据"""
        try:
            text = item.get("text", "").strip()
            if not text:
                return None

            # 简单的表格解析逻辑
            lines = text.split('\n')
            if len(lines) < 2:
                return None

            # 检查是否是表格格式（包含|字符）
            if '|' not in lines[0]:
                return None

            # 解析表格
            headers = [h.strip() for h in lines[0].split('|') if h.strip()]
            rows = []

            for line in lines[1:]:
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(row) == len(headers):
                        rows.append(row)

            if not rows:
                return None

            return {
                'id': f'table_{idx}',
                'title': f'表格 {idx + 1}',
                'headers': headers,
                'rows': rows,
                'rowCount': len(rows),
                'columnCount': len(headers),
                'confidence': 90.0,
                'bbox': item.get("bbox", []),
                'page': item.get("page_idx", 0)
            }

        except Exception as e:
            logger.error(f"表格提取失败 {idx}: {str(e)}")
            return None

    def _extract_image_from_item(self, item: Dict[str, Any], images_data: Dict[str, str], idx: int) -> Optional[Dict[str, Any]]:
        """从content_list条目提取图片数据"""
        try:
            img_path = item.get("img_path", "")
            text = item.get("text", "").strip()

            # 检查是否有对应的base64图片数据
            img_base64 = None
            if images_data:
                logger.info(f"🖼️  图片 {idx}: img_path={img_path}, images_data有{len(images_data)}个图片")
                if img_path in images_data:
                    img_base64 = images_data[img_path]
                    logger.info(f"   ✅ 找到base64数据，长度: {len(img_base64) if img_base64 else 0}")
                else:
                    logger.warning(f"   ⚠️  未找到base64数据，可用的key: {list(images_data.keys())[:3]}")
            else:
                logger.warning(f"🖼️  图片 {idx}: images_data为空")

            return {
                'id': f'image_{idx}',
                'type': '图像',
                'path': img_path,
                'base64': img_base64,
                'altText': text,
                'description': text if text else f"图片 {idx + 1}",
                'confidence': 88.0,
                'bbox': item.get("bbox", []),
                'page': item.get("page_idx", 0)
            }

        except Exception as e:
            logger.error(f"图片提取失败 {idx}: {str(e)}")
            return None

    def _extract_formula_from_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """从content_list条目提取公式数据"""
        try:
            text = item.get("text", "").strip()
            if not text:
                return None

            # 判断是行内公式还是块公式
            is_block = len(text) > 20 or '\n' in text
            formula_type = "block" if is_block else "inline"

            return {
                'id': f'formula_{idx}',
                'type': formula_type,
                'formula': text,
                'description': f"数学公式 ({formula_type})",
                'confidence': 85.0,
                'bbox': item.get("bbox", []),
                'page': item.get("page_idx", 0)
            }

        except Exception as e:
            logger.error(f"公式提取失败 {idx}: {str(e)}")
            return None

    def _extract_text_block_from_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """从content_list条目提取文本块"""
        try:
            text = item.get("text", "").strip()
            if not text:
                return None

            # 判断文本类型
            item_type = item.get("type", "")
            if "title" in item_type.lower():
                text_type = "heading"
            elif "paragraph" in item_type.lower():
                text_type = "paragraph"
            else:
                text_type = "text"

            return {
                'id': f'text_{idx}',
                'type': text_type,
                'title': text[:50] + "..." if len(text) > 50 else text,
                'content': text,
                'level': 0,
                'bbox': item.get("bbox", []),
                'page': item.get("page_idx", 0)
            }

        except Exception as e:
            logger.error(f"文本块提取失败 {idx}: {str(e)}")
            return None

    def _extract_keywords_from_markdown(self, content: str) -> List[str]:
        """从markdown内容中提取���键词"""
        # 简单的关键词提取逻辑
        # 可以使用更复杂的NLP方法
        common_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        # 提取中文词汇（简单实现）
        words = re.findall(r'[\u4e00-\u9fff]+', content)
        word_freq = {}

        for word in words:
            if len(word) >= 2 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # 返回频率最高的10个词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]

    def parse(self, content: str) -> Dict[str, Any]:
        """
        Parse markdown content and extract structured data (传统方法)

        Args:
            content: Markdown content string

        Returns:
            Structured content dictionary with extracted elements
        """

        try:
            # Clean content
            content = self._clean_content(content)

            # Extract different content types
            tables = self._extract_tables(content)
            formulas = self._extract_formulas(content)
            images = self._extract_images(content)
            text_blocks = self._extract_text_blocks(content)

            # Extract keywords from full text
            keywords = self._extract_keywords_from_markdown(content)

            return {
                'text': {
                    'fullText': content,
                    'textBlocks': text_blocks,
                    'keywords': keywords,
                    'confidence': 90.0
                },
                'tables': tables,
                'formulas': formulas,
                'images': images,
                'handwritten': {
                    'detected': False,
                    'text': '',
                    'confidence': 0,
                    'areas': []
                },
                'performance': {
                    'accuracy': 95.0,
                    'speed': 2.1,
                    'memory': 384
                },
                'metadata': {
                    'totalElements': len(text_blocks) + len(tables) + len(formulas) + len(images),
                    'contentTypes': ['text'],
                    'processingTime': None
                }
            }

        except Exception as e:
            logger.error(f"Failed to parse markdown content: {str(e)}")
            raise

    def _clean_content(self, content: str) -> str:
        """Clean markdown content"""
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()

    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown content"""
        tables = []
        matches = self.patterns['table'].finditer(content)

        for idx, match in enumerate(matches):
            header_line = match.group(1).strip()
            table_content = match.group(2)

            # Parse headers
            headers = [h.strip() for h in header_line.split('|') if h.strip()]

            # Parse rows
            rows = []
            for line in table_content.strip().split('\n'):
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(row) == len(headers):
                        rows.append(row)

            if headers and rows:
                tables.append({
                    'id': f'table_{idx}',
                    'title': f'表格 {idx + 1}',
                    'headers': headers,
                    'rows': rows,
                    'rowCount': len(rows),
                    'columnCount': len(headers),
                    'confidence': 90.0
                })

        return tables

    def _extract_formulas(self, content: str) -> List[Dict[str, Any]]:
        """Extract formulas from markdown content"""
        formulas = []

        # Block formulas
        block_matches = self.patterns['block_formula'].finditer(content)
        for idx, match in enumerate(block_matches):
            formula_content = match.group(1).strip()
            formulas.append({
                'id': f'block_formula_{idx}',
                'type': 'block',
                'formula': formula_content,
                'description': '块级公式',
                'confidence': 85.0
            })

        # Inline formulas
        inline_matches = self.patterns['inline_formula'].finditer(content)
        inline_offset = len(formulas)
        for idx, match in enumerate(inline_matches):
            formula_content = match.group(1).strip()
            formulas.append({
                'id': f'inline_formula_{inline_offset + idx}',
                'type': 'inline',
                'formula': formula_content,
                'description': '行内公式',
                'confidence': 80.0
            })

        return formulas

    def _extract_images(self, content: str) -> List[Dict[str, Any]]:
        """Extract images from markdown content"""
        images = []
        matches = self.patterns['image'].finditer(content)

        for idx, match in enumerate(matches):
            alt_text = match.group(1).strip()
            image_path = match.group(2).strip()

            images.append({
                'id': f'image_{idx}',
                'type': '图像',
                'path': image_path,
                'altText': alt_text,
                'description': alt_text if alt_text else f'图片 {idx + 1}',
                'confidence': 90.0
            })

        return images

    def _extract_text_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract structured text blocks from content"""
        text_blocks = []

        # Split content into sections
        sections = re.split(r'\n(#{1,6})\s+', content)

        current_section = "正文"
        current_content = sections[0] if sections else ""

        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                # Save previous section
                if current_content.strip():
                    text_blocks.append({
                        'type': 'section',
                        'title': current_section,
                        'content': current_content.strip(),
                        'level': 0
                    })

                # Start new section
                current_section = sections[i].strip('#').strip()
                current_content = sections[i + 1]

        # Save last section
        if current_content.strip():
            text_blocks.append({
                'type': 'section',
                'title': current_section,
                'content': current_content.strip(),
                'level': 0
            })

        return text_blocks

    def _extract_html_tables(self, markdown_content: str) -> List[Dict[str, Any]]:
        """从markdown中提取HTML表格"""
        try:
            import re

            # 查找所有HTML表格
            table_pattern = r'<table[^>]*>(.*?)</table>'
            tables = re.findall(table_pattern, markdown_content, re.DOTALL | re.IGNORECASE)

            extracted_tables = []
            for idx, table_html in enumerate(tables):
                # 提取表格行和单元格
                row_pattern = r'<tr[^>]*>(.*?)</tr>'
                rows = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

                if not rows:
                    continue

                # 解析表头和数据行
                table_data = []
                headers = []

                for row_idx, row_html in enumerate(rows):
                    # 提取单元格 (th或td)
                    cell_pattern = r'<t[hd][^>]*>(.*?)</t[hd]>'
                    cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

                    # 清理单元格内容
                    cleaned_cells = []
                    for cell in cells:
                        # 移除HTML标签和多余空格，解码HTML实体
                        clean_text = re.sub(r'<[^>]+>', '', cell).strip()
                        # 解码HTML实体如 &#x27;
                        import html
                        clean_text = html.unescape(clean_text)
                        cleaned_cells.append(clean_text)

                    if cleaned_cells:
                        if row_idx == 0:
                            # 第一行作为表头
                            headers = cleaned_cells
                        else:
                            table_data.append(cleaned_cells)

                if headers and table_data:
                    extracted_tables.append({
                        'id': f'html_table_{idx}',
                        'title': f'表格 {idx + 1}',
                        'headers': headers,
                        'rows': table_data,
                        'rowCount': len(table_data),
                        'columnCount': len(headers),
                        'confidence': 85.0,
                        'source': 'html_markdown'
                    })

            logger.info(f"✅ 从HTML中提取了 {len(extracted_tables)} 个表格")
            return extracted_tables

        except Exception as e:
            logger.error(f"HTML表格提取失败: {str(e)}")
            return []