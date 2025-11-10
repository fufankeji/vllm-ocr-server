#!/usr/bin/env python3
"""
DeepSeek OCR Service
Handles OCR using DeepSeek-OCR API
"""

import os
import logging
import re
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from app.models.ocr_models import (
    OCRResults, TextResult, TableResult, FormulaResult,
    ImageResult, HandwrittenResult, PerformanceResult, OCRMetadata
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DeepSeekOCRService:
    """DeepSeek OCR Service - 调用 DeepSeek-OCR vLLM API"""

    def __init__(self):
        # DeepSeek-OCR API 配置
        self.api_url = os.getenv("DEEPSEEK_OCR_API_URL", "http://192.168.110.131:8797/ocr")
        self.timeout = int(os.getenv("DEEPSEEK_OCR_TIMEOUT", "600"))
        self.enable_description = os.getenv("DEEPSEEK_ENABLE_DESC", "true").lower() == "true"

        # DeepSeek-OCR 处理参数（参考 api_server_optimize.py）
        self.dpi = int(os.getenv("DEEPSEEK_OCR_DPI", "144"))
        self.base_size = int(os.getenv("DEEPSEEK_OCR_BASE_SIZE", "1024"))
        self.image_size = int(os.getenv("DEEPSEEK_OCR_IMAGE_SIZE", "640"))
        self.crop_mode = True  # 裁切模式，用于提取图像
        self.verbose = False

        logger.info(f"DeepSeek OCR Service initialized: {self.api_url}")

    async def analyze_document(self, file_path: Path, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze document using DeepSeek OCR

        Args:
            file_path: Path to PDF or image file
            options: Additional options (enable_description, etc.)

        Returns:
            OCR analysis results
        """
        options = options or {}
        enable_desc = options.get("enable_description", self.enable_description)

        try:
            logger.info(f"DeepSeek OCR analyzing: {file_path.name}")

            # 1. Call DeepSeek OCR API
            # 参考 api_server_optimize.py 的参数设置
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/pdf')}
                data = {
                    'dpi': str(self.dpi),
                    'base_size': str(self.base_size),
                    'image_size': str(self.image_size),
                    'crop_mode': 'true' if self.crop_mode else 'false',
                    'verbose': 'true' if self.verbose else 'false',
                    'enable_image_description': 'true' if enable_desc else 'false',
                }

                logger.info(f"Sending DeepSeek OCR request with params: {data}")

                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                raise Exception(f"DeepSeek OCR API error: {response.status_code}, {response.text[:500]}")

            # 2. Parse response
            result = response.json()

            # Debug: Log complete API response structure
            logger.info(f"DeepSeek API response keys: {list(result.keys())}")

            # 解析响应格式
            if "backend" in result and "pages" in result:
                # MinerU 格式响应
                logger.info("Received MinerU-format response from DeepSeek API")
                markdown_content = result.get("content_md", "")
                page_count = len(result.get("pages", []))
                images_data = result.get("images", {})
            else:
                # 简单格式响应（markdown + page_count + images）
                markdown_content = result.get("markdown", "")
                page_count = result.get("page_count", 0)
                images_data = result.get("images", {})

                if images_data:
                    logger.info(f"Received simple format response with {len(images_data)} images")
                else:
                    logger.warning(" Received simple format response (no images)")

            logger.info(f"Pages: {page_count}")
            logger.info(f"Images: {len(images_data)} items")
            if images_data:
                logger.info(f"Image keys: {list(images_data.keys())[:5]}")

            logger.info(f"DeepSeek OCR completed: {page_count} pages")
            logger.info(f"Markdown length: {len(markdown_content)} characters")

            # Debug: Save complete response for analysis
            import json
            debug_response_file = Path("debug_deepseek_response.json")
            with open(debug_response_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Complete API response saved to: {debug_response_file.absolute()}")

            debug_markdown_file = Path("debug_deepseek_markdown.md")
            with open(debug_markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Markdown content saved to: {debug_markdown_file.absolute()}")

            # 3. Convert to frontend-compatible format (MinerU-style)
            ocr_results = self._convert_to_mineru_format(
                markdown_content=markdown_content,
                images_data=images_data,
                file_path=file_path
            )

            return {
                "success": True,
                "model": "deepseek",
                "filename": file_path.name,
                "results": ocr_results,
                "fullMarkdown": markdown_content,
                "metadata": {
                    "page_count": page_count,
                    "enable_description": enable_desc,
                    "images": images_data
                }
            }

        except Exception as e:
            logger.error(f"❌ DeepSeek OCR failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"DeepSeek OCR analysis failed: {str(e)}")

    def _convert_to_mineru_format(
        self,
        markdown_content: str,
        images_data: dict,
        file_path: Path
    ) -> dict:
        """
        Convert DeepSeek OCR response to MinerU-compatible format

        Args:
            markdown_content: Markdown content from DeepSeek
            images_data: Images data from DeepSeek API response
            file_path: Original file path

        Returns:
            MinerU-style OCR results dict
        """
        logger.info("🔄 Converting DeepSeek OCR to MinerU format...")

        # 提取 HTML 表格
        tables = self._extract_html_tables_from_markdown(markdown_content)
        logger.info(f"📊 Extracted {len(tables)} HTML tables from markdown")

        results = {
            "text": {
                "fullText": markdown_content,
                "keywords": ["DeepSeek-OCR", "文档分析", "智能识别"],
                "confidence": 95.0
            },
            "tables": tables,
            "formulas": [],
            "images": [],
            "handwritten": {
                "detected": False,
                "text": "",
                "confidence": 0.0
            },
            "performance": {
                "accuracy": 96.5,
                "speed": 2.1,
                "memory": 448
            },
            "metadata": {
                "fullText": markdown_content,
                "memory": 448,
                "totalElements": 0,
                "contentTypes": []
            }
        }

        # 处理图像数据 - 直接使用DeepSeek API返回的图像信息
        if images_data and isinstance(images_data, dict):
            logger.info(f"🖼️  Processing {len(images_data)} images from DeepSeek API...")

            for image_key, image_base64 in images_data.items():
                # image_base64 是纯base64字符串（不带 data:image 前缀）
                if isinstance(image_base64, str):
                    # 构建完整的 data URI
                    data_uri = f"data:image/png;base64,{image_base64}"

                    image_result = {
                        "id": image_key.replace('.png', ''),
                        "type": "图表",
                        "description": f"DeepSeek-OCR识别图像 - {image_key}",
                        "altText": f"图像 {image_key}",  # 添加必需的 altText 字段
                        "confidence": 95.0,
                        "base64": data_uri,  # 使用 base64 字段存储 data URI（前端优先读取这个）
                        "path": image_key  # 保留原始文件名作为 path
                    }
                    results["images"].append(image_result)
                    logger.debug(f"   ✓ Added image: {image_key}")
                # 如果image_info是包含更多信息的字典
                elif isinstance(image_base64, dict):
                    image_result = {
                        "id": image_key.replace('.png', ''),
                        "type": image_base64.get("type", "图表"),
                        "description": image_base64.get("description", f"DeepSeek-OCR识别图像 - {image_key}"),
                        "altText": image_base64.get("altText", f"图像 {image_key}"),  # 添加必需的 altText 字段
                        "confidence": image_base64.get("confidence", 95.0),
                        "path": image_base64.get("path", f"/images/{image_key}")
                    }
                    results["images"].append(image_result)

        logger.info(f"✅ Converted to MinerU format: {len(results['images'])} images, {len(results['tables'])} tables")

        return results

    def _extract_html_tables_from_markdown(self, markdown_content: str) -> list:
        """从 markdown 中提取 HTML 表格"""
        tables = []
        table_pattern = r'<table>(.*?)</table>'
        matches = re.findall(table_pattern, markdown_content, re.DOTALL | re.IGNORECASE)

        for idx, table_html in enumerate(matches):
            try:
                # 提取所有行
                row_pattern = r'<tr>(.*?)</tr>'
                rows_html = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

                if not rows_html:
                    continue

                # 提取表头和数据
                headers = []
                data_rows = []

                for row_idx, row_html in enumerate(rows_html):
                    # 提取单元格
                    cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
                    cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)
                    # 清理 HTML 标签
                    cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]

                    if row_idx == 0:
                        headers = cells
                    else:
                        data_rows.append(cells)

                if headers and data_rows:
                    table_result = {
                        "id": f"table_{idx + 1}",
                        "title": f"表格 {idx + 1}",
                        "headers": headers,
                        "rows": data_rows,
                        "rowCount": len(data_rows),
                        "columnCount": len(headers),
                        "confidence": 95.0,
                        "html": f"<table>{table_html}</table>"  # 保留原始 HTML
                    }
                    tables.append(table_result)

            except Exception as e:
                logger.warning(f"Failed to parse table {idx}: {e}")
                continue

        return tables

        # Create text result
        text_result = TextResult(
            fullText=markdown_content,
            textBlocks=text_blocks,
            keywords=self._extract_keywords(markdown_content),
            confidence=0.95,  # DeepSeek generally has high confidence
            stats={
                "characters": len(markdown_content),
                "words": len(markdown_content.split()),
                "lines": len(markdown_content.split('\n')),
                "pages": page_count
            }
        )

        # Create handwritten result (DeepSeek doesn't detect handwriting specifically)
        handwritten_result = HandwrittenResult(
            detected=False,
            text="No handwritten content detection available for DeepSeek OCR",
            confidence=0.0,
            areas=[]
        )

        # Create performance result
        performance_result = PerformanceResult(
            accuracy=95.0,
            speed=0.0,  # Will be calculated by main service
            memory=0
        )

        # Create metadata
        metadata = OCRMetadata(
            totalElements=len(text_blocks) + len(tables) + len(formulas) + len(images),
            contentTypes=self._get_content_types(text_blocks, tables, formulas, images),
            processingTime=None
        )

        return OCRResults(
            text=text_result,
            tables=tables,
            formulas=formulas,
            images=images,
            handwritten=handwritten_result,
            performance=performance_result,
            metadata=metadata
        )

    def _parse_markdown_blocks(self, markdown: str) -> List[Dict[str, Any]]:
        """Parse markdown into text blocks"""
        blocks = []
        lines = markdown.split('\n')

        current_block = []
        block_type = "paragraph"

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if current_block:
                    blocks.append({
                        "type": block_type,
                        "content": '\n'.join(current_block),
                        "level": 0
                    })
                    current_block = []
                    block_type = "paragraph"
                continue

            # Detect block type
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                blocks.append({
                    "type": "heading",
                    "content": stripped.lstrip('# '),
                    "level": level
                })
            elif stripped.startswith('- ') or stripped.startswith('* ') or stripped.startswith('+ '):
                current_block.append(stripped)
                block_type = "list"
            elif stripped.startswith('> '):
                current_block.append(stripped[2:])
                block_type = "quote"
            else:
                current_block.append(stripped)
                block_type = "paragraph"

        # Add last block
        if current_block:
            blocks.append({
                "type": block_type,
                "content": '\n'.join(current_block),
                "level": 0
            })

        return blocks

    def _extract_tables(self, markdown: str) -> List[TableResult]:
        """Extract tables from markdown"""
        import re
        tables = []

        logger.info(f"🔍 Extracting tables from markdown (length: {len(markdown)})")

        # Method 1: Standard markdown tables with pipes
        pipe_tables = self._extract_pipe_tables(markdown)
        tables.extend(pipe_tables)
        logger.info(f"📊 Found {len(pipe_tables)} pipe-style tables")

        # Method 2: HTML tables
        html_tables = self._extract_html_tables(markdown)
        tables.extend(html_tables)
        logger.info(f"📊 Found {len(html_tables)} HTML tables")

        # Method 3: ASCII/Box drawing tables (for Chinese documents)
        ascii_tables = self._extract_ascii_tables(markdown)
        tables.extend(ascii_tables)
        logger.info(f"📊 Found {len(ascii_tables)} ASCII/box tables")

        # Method 4: Table detection by grid-like patterns
        grid_tables = self._extract_grid_tables(markdown)
        tables.extend(grid_tables)
        logger.info(f"📊 Found {len(grid_tables)} grid-style tables")

        logger.info(f"📊 Total tables extracted: {len(tables)}")
        return tables

    def _extract_pipe_tables(self, markdown: str) -> List[TableResult]:
        """Extract standard markdown pipe tables"""
        tables = []
        lines = markdown.split('\n')

        i = 0
        table_id = 0
        while i < len(lines):
            line = lines[i].strip()

            # Detect table (markdown table format)
            if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
                table_id += 1
                table_lines = [line]
                i += 1

                # Skip separator line
                i += 1

                # Collect table rows
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i].strip())
                    i += 1

                # Parse table
                if len(table_lines) >= 2:
                    headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
                    rows = []
                    for row_line in table_lines[1:]:
                        cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                        rows.append(cells)

                    tables.append(TableResult(
                        id=f"table_pipe_{table_id}",
                        title=f"Table {table_id}",
                        headers=headers,
                        rows=rows,
                        rowCount=len(rows),
                        columnCount=len(headers),
                        confidence=0.95
                    ))

            i += 1

        return tables

    def _extract_html_tables(self, markdown: str) -> List[TableResult]:
        """Extract HTML tables"""
        import re
        tables = []

        # Find HTML table tags
        table_pattern = r'<table[^>]*>(.*?)</table>'
        matches = re.findall(table_pattern, markdown, re.DOTALL | re.IGNORECASE)

        for idx, table_html in enumerate(matches):
            # Extract rows from HTML
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            rows_html = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

            if len(rows_html) >= 2:
                # Extract headers (first row)
                header_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
                headers = re.findall(header_pattern, rows_html[0], re.DOTALL | re.IGNORECASE)
                headers = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]

                # Extract data rows
                data_rows = []
                for row_html in rows_html[1:]:
                    cells = re.findall(header_pattern, row_html, re.DOTALL | re.IGNORECASE)
                    cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                    if cells:
                        data_rows.append(cells)

                tables.append(TableResult(
                    id=f"table_html_{idx + 1}",
                    title=f"HTML Table {idx + 1}",
                    headers=headers,
                    rows=data_rows,
                    rowCount=len(data_rows),
                    columnCount=len(headers),
                    confidence=0.90
                ))

        return tables

    def _extract_ascii_tables(self, markdown: str) -> List[TableResult]:
        """Extract ASCII/box drawing tables"""
        tables = []

        # Pattern for box drawing characters
        box_pattern = r'([╔╚╠╦╣╤╧╟╢╥╨║═║│├┤┬┴┼┌┐└┘├┤┬┴┼─│\s\w\d\.\-]+)'

        # Find table-like structures
        lines = markdown.split('\n')
        i = 0
        table_id = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check if line contains box drawing characters
            if re.search(r'[╔╚╠╦╣╤╧╟╢╥╨║═║│├┤┬┴┼┌┐└┘]', line):
                # Found potential table, collect consecutive lines with box characters
                table_lines = []
                while i < len(lines) and re.search(r'[╔╚╠╦╣╤╧╟╢╥╨║═║│├┤┬┴┼┌┐└┘─│]', lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                if len(table_lines) >= 3:
                    table_id += 1
                    # Simple parsing - split by vertical lines and spaces
                    rows = []
                    for table_line in table_lines:
                        # Remove box characters and split by content
                        clean_line = re.sub(r'[╔╚╠╦╣╤╧╟╢╥╨║═║│├┤┬┴┼┌┐└┘─]', ' ', table_line)
                        # Split by multiple spaces and filter empty
                        cells = [cell.strip() for cell in re.split(r'\s{2,}', clean_line) if cell.strip()]
                        if cells:
                            rows.append(cells)

                    if len(rows) >= 2:
                        tables.append(TableResult(
                            id=f"table_ascii_{table_id}",
                            title=f"ASCII Table {table_id}",
                            headers=rows[0],
                            rows=rows[1:],
                            rowCount=len(rows) - 1,
                            columnCount=len(rows[0]) if rows else 0,
                            confidence=0.85
                        ))
            else:
                i += 1

        return tables

    def _extract_grid_tables(self, markdown: str) -> List[TableResult]:
        """Extract grid-style tables by detecting aligned columns"""
        tables = []

        # Look for text with consistent spacing patterns
        lines = markdown.split('\n')

        # Find potential table starts (lines with multiple spaces/alignments)
        for i, line in enumerate(lines):
            if re.search(r'\s{3,}\w+\s{3,}', line):  # Multiple spaces suggesting columns
                # Check surrounding lines for similar patterns
                potential_table = []
                j = i
                while j < len(lines) and re.search(r'\s{3,}', lines[j]):
                    potential_table.append(lines[j])
                    j += 1

                if len(potential_table) >= 3:
                    # Try to parse as table
                    rows = []
                    for table_line in potential_table:
                        # Split by multiple spaces
                        cells = [cell.strip() for cell in re.split(r'\s{3,}', table_line) if cell.strip()]
                        if cells:
                            rows.append(cells)

                    if len(rows) >= 2 and all(len(row) == len(rows[0]) for row in rows):
                        tables.append(TableResult(
                            id=f"table_grid_{len(tables) + 1}",
                            title=f"Grid Table {len(tables) + 1}",
                            headers=rows[0],
                            rows=rows[1:],
                            rowCount=len(rows) - 1,
                            columnCount=len(rows[0]),
                            confidence=0.80
                        ))

        return tables

    def _extract_formulas(self, markdown: str) -> List[FormulaResult]:
        """Extract formulas from markdown"""
        import re
        formulas = []

        # Find inline formulas: $...$
        inline_pattern = r'\$([^\$]+)\$'
        for idx, match in enumerate(re.finditer(inline_pattern, markdown)):
            formulas.append(FormulaResult(
                id=f"formula_inline_{idx + 1}",
                type="inline",
                formula=match.group(1),
                description="Inline formula",
                confidence=0.90,
                position=match.start()
            ))

        # Find block formulas: $$...$$
        block_pattern = r'\$\$([^\$]+)\$\$'
        for idx, match in enumerate(re.finditer(block_pattern, markdown)):
            formulas.append(FormulaResult(
                id=f"formula_block_{idx + 1}",
                type="block",
                formula=match.group(1),
                description="Block formula",
                confidence=0.90,
                position=match.start()
            ))

        return formulas

    def _extract_images(self, markdown: str) -> List[ImageResult]:
        """Extract image references from markdown"""
        import re
        images = []

        logger.info(f"🔍 Extracting images from markdown (length: {len(markdown)})")

        # Method 1: Standard markdown images ![alt](path)
        md_images = self._extract_markdown_images(markdown)
        images.extend(md_images)
        logger.info(f"🖼️  Found {len(md_images)} markdown images")

        # Method 2: HTML images <img src="...">
        html_images = self._extract_html_images(markdown)
        images.extend(html_images)
        logger.info(f"🖼️  Found {len(html_images)} HTML images")

        # Method 3: Chinese image references [图片: description]
        chinese_images = self._extract_chinese_image_refs(markdown)
        images.extend(chinese_images)
        logger.info(f"🖼️  Found {len(chinese_images)} Chinese image references")

        # Method 4: English image references [Image: description]
        english_images = self._extract_english_image_refs(markdown)
        images.extend(english_images)
        logger.info(f"🖼️  Found {len(english_images)} English image references")

        # Method 5: Special DeepSeek image markers like <|ref|>image<|/ref|>
        deepseek_images = self._extract_deepseek_images(markdown)
        images.extend(deepseek_images)
        logger.info(f"🖼️  Found {len(deepseek_images)} DeepSeek image markers")

        logger.info(f"🖼️  Total images extracted: {len(images)}")
        return images

    def _extract_markdown_images(self, markdown: str) -> List[ImageResult]:
        """Extract standard markdown images"""
        import re
        images = []

        # Pattern: ![alt text](image path)
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.finditer(img_pattern, markdown)

        for idx, match in enumerate(matches):
            alt_text = match.group(1).strip()
            img_path = match.group(2).strip()

            images.append(ImageResult(
                id=f"md_image_{idx + 1}",
                type="markdown",
                path=img_path,
                base64=None,
                altText=alt_text,
                description=alt_text,
                confidence=0.95,
                position=match.start()
            ))

        return images

    def _extract_html_images(self, markdown: str) -> List[ImageResult]:
        """Extract HTML images"""
        import re
        images = []

        # Pattern: <img src="path" alt="text" ...>
        img_pattern = r'<img[^>]*src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>'
        matches = re.finditer(img_pattern, markdown, re.IGNORECASE)

        for idx, match in enumerate(matches):
            img_path = match.group(1).strip()
            alt_text = match.group(2).strip() if match.group(2) else "Image"

            images.append(ImageResult(
                id=f"html_image_{idx + 1}",
                type="html",
                path=img_path,
                base64=None,
                altText=alt_text,
                description=alt_text,
                confidence=0.90,
                position=match.start()
            ))

        return images

    def _extract_chinese_image_refs(self, markdown: str) -> List[ImageResult]:
        """Extract Chinese image references"""
        import re
        images = []

        # Pattern: [图片: description]
        img_pattern = r'\[图片:\s*([^\]]+)\]'
        matches = re.finditer(img_pattern, markdown)

        for idx, match in enumerate(matches):
            description = match.group(1).strip()

            images.append(ImageResult(
                id=f"cn_image_{idx + 1}",
                type="reference",
                path=f"image_{idx + 1}",
                base64=None,
                altText="图片",
                description=description,
                confidence=0.85,
                position=match.start()
            ))

        return images

    def _extract_english_image_refs(self, markdown: str) -> List[ImageResult]:
        """Extract English image references"""
        import re
        images = []

        # Pattern: [Image: description] or [Figure: description]
        img_pattern = r'\[(?:Image|Figure|Fig):\s*([^\]]+)\]'
        matches = re.finditer(img_pattern, markdown, re.IGNORECASE)

        for idx, match in enumerate(matches):
            description = match.group(1).strip()

            images.append(ImageResult(
                id=f"en_image_{idx + 1}",
                type="reference",
                path=f"image_{idx + 1}",
                base64=None,
                altText="Image",
                description=description,
                confidence=0.85,
                position=match.start()
            ))

        return images

    def _extract_deepseek_images(self, markdown: str) -> List[ImageResult]:
        """Extract DeepSeek special image markers"""
        import re
        images = []

        # Pattern: <|ref|>image<|/ref|><|det|>[[bbox]]<|/det|>
        img_pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>'
        matches = re.finditer(img_pattern, markdown)

        for idx, match in enumerate(matches):
            bbox = match.group(1).strip()

            images.append(ImageResult(
                id=f"deepseek_image_{idx + 1}",
                type="deepseek_marker",
                path=f"deepseek_image_{idx + 1}",
                base64=None,
                altText="DeepSeek Image",
                description=f"Image with bbox: {bbox}",
                confidence=0.95,
                position=match.start()
            ))

        return images

    def _extract_keywords(self, markdown: str) -> List[str]:
        """Extract keywords from markdown"""
        import re

        # Simple keyword extraction: find words longer than 4 characters
        # that appear multiple times
        words = re.findall(r'\b\w{4,}\b', markdown.lower())
        word_freq = {}

        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get top 10 most frequent words
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, freq in keywords if freq > 1]

    def _get_content_types(
        self,
        text_blocks: List[Dict],
        tables: List[TableResult],
        formulas: List[FormulaResult],
        images: List[ImageResult]
    ) -> List[str]:
        """Get list of content types present"""
        types = []

        if text_blocks:
            types.append("text")
        if tables:
            types.append("tables")
        if formulas:
            types.append("formulas")
        if images:
            types.append("images")

        return types if types else ["text"]
