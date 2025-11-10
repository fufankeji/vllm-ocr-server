"""
PaddleOCR-VL 服务封装
处理 PDF 和图片文件的 OCR 识别
"""

import base64
import json
import logging
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
import re

logger = logging.getLogger(__name__)


class PaddleOCRService:
    """PaddleOCR-VL OCR 服务"""

    def __init__(self):
        """
        初始化 PaddleOCR 服务
        """
        self.api_url = os.getenv("PADDLEOCR_API_URL", "http://192.168.110.131:10800/layout-parsing")
        logger.info(f"🔧 Initialized PaddleOCR service with API: {self.api_url}")

    async def process_file(self, file_path: str) -> Dict:
        """
        处理文件（PDF 或图片）

        Args:
            file_path: 文件路径

        Returns:
            包含 markdown、images、tables、formulas 的结果字典
        """
        try:
            logger.info(f"Processing file with PaddleOCR: {file_path}")

            # 读取文件并编码为 base64
            with open(file_path, "rb") as f:
                file_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 判断文件类型
            file_extension = Path(file_path).suffix.lower()
            if file_extension == ".pdf":
                file_type = 0  # PDF
            elif file_extension in [".png", ".jpg", ".jpeg", ".bmp"]:
                file_type = 1  # 图片
            else:
                logger.warning(f"⚠️  Unknown file type: {file_extension}, treating as image")
                file_type = 1

            # 构造请求
            payload = {
                "file": file_base64,
                "fileType": file_type,
                "prettifyMarkdown": True,
                "visualize": False,
            }

            headers = {"Content-Type": "application/json"}

            # 发送请求
            logger.info("Sending request to PaddleOCR API...")
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=300
            )

            if response.status_code != 200:
                error_msg = f"PaddleOCR API error: {response.status_code}"
                logger.error(f"{error_msg}")
                logger.error(f"Response: {response.text[:500]}")
                raise Exception(error_msg)

            # 解析响应
            result = response.json()

            # 检查错误码
            error_code = result.get("errorCode")
            if error_code != 0:
                error_msg = result.get("errorMsg", "Unknown error")
                logger.error(f"PaddleOCR service error (code: {error_code}): {error_msg}")
                raise Exception(f"PaddleOCR error: {error_msg}")

            logger.info("✅ PaddleOCR API response received")

            # 提取和处理结果
            processed_result = self._process_response(result, file_path)

            return processed_result

        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}")
            raise

    def _process_response(self, api_response: Dict, file_path: str) -> Dict:
        """
        处理 PaddleOCR API 响应

        Args:
            api_response: API 原始响应
            file_path: 原始文件路径

        Returns:
            标准化的结果字典
        """
        logger.info("Processing PaddleOCR response...")

        result_data = api_response.get("result", {})
        layout_parsing_results = result_data.get("layoutParsingResults", [])

        # 初始化结果
        results = {
            "markdown": "",
            "images": [],
            "tables": [],
            "formulas": [],
            "metadata": {
                "total_pages": len(layout_parsing_results),
                "file_name": Path(file_path).name,
            }
        }

        # 收集所有页面的内容
        all_markdown_parts = []

        for page_idx, page_result in enumerate(layout_parsing_results):
            logger.info(f"Processing page {page_idx + 1}...")

            # 提取 markdown
            markdown_data = page_result.get("markdown", {})
            markdown_text = markdown_data.get("text", "")
            # 注意：字段名是 "images" 不是 "markdownImages"
            markdown_images = markdown_data.get("images", {})

            # 添加页面标记
            all_markdown_parts.append(f"\n\n# Page {page_idx + 1}\n\n")
            all_markdown_parts.append(markdown_text)

            # 从 markdown_images 字典提取图片
            if markdown_images:
                logger.info(f"   Found {len(markdown_images)} images in markdown.images")
                for img_filename, img_base64 in markdown_images.items():
                    # 构建 data URI
                    # 判断图片格式
                    if img_filename.endswith('.jpg') or img_filename.endswith('.jpeg'):
                        data_uri = f"data:image/jpeg;base64,{img_base64}"
                    elif img_filename.endswith('.png'):
                        data_uri = f"data:image/png;base64,{img_base64}"
                    else:
                        data_uri = f"data:image/jpeg;base64,{img_base64}"  # 默认 jpeg

                    image_info = {
                        "id": f"page_{page_idx}_{img_filename.replace('.jpg', '').replace('.png', '')}",
                        "type": "图片",
                        "description": f"Page {page_idx + 1} - {img_filename}",
                        "altText": img_filename,
                        "confidence": 95.0,
                        "base64": data_uri,
                        "path": img_filename,
                        "page": page_idx
                    }
                    results["images"].append(image_info)

            # 从 markdown 文本中提取 HTML 表格
            tables_in_page = self._extract_tables_from_markdown(markdown_text, page_idx)
            results["tables"].extend(tables_in_page)

        # 合并所有页面的 markdown
        results["markdown"] = "".join(all_markdown_parts)

        logger.info(f"✅ Extracted: {len(results['images'])} images, "
                   f"{len(results['tables'])} tables, "
                   f"{len(results['formulas'])} formulas")

        return results

    def _extract_image_from_region(
        self,
        region: Dict,
        page_idx: int,
        markdown_images: Dict
    ) -> Optional[Dict]:
        """从 region 中提取图片信息"""
        try:
            # 获取图片的 bbox
            bbox = region.get("bbox", [])
            if len(bbox) < 4:
                return None

            # 查找对应的 base64 图片数据
            # markdown 中的图片路径格式: imgs/img_in_image_box_x1_y1_x2_y2.jpg
            image_filename = None
            image_base64 = None

            # 从 markdownImages 中查找匹配的图片
            for img_name, img_data in markdown_images.items():
                if f"_{int(bbox[0])}_{int(bbox[1])}_" in img_name:
                    image_filename = img_name
                    image_base64 = img_data
                    break

            if not image_base64:
                logger.warning(f"No base64 data found for image at bbox {bbox}")
                return None

            # 构建 data URI
            data_uri = f"data:image/jpeg;base64,{image_base64}"

            return {
                "id": f"page_{page_idx}_img_{int(bbox[0])}_{int(bbox[1])}",
                "type": region.get("type", "图片"),
                "description": f"Page {page_idx + 1} - {region.get('type', 'Image')}",
                "altText": image_filename or f"Image on page {page_idx + 1}",
                "confidence": 95.0,
                "base64": data_uri,
                "path": image_filename,
                "bbox": bbox,
                "page": page_idx
            }

        except Exception as e:
            logger.warning(f"Failed to extract image from region: {e}")
            return None

    def _extract_table_from_region(
        self,
        region: Dict,
        page_idx: int
    ) -> Optional[Dict]:
        """从 region 中提取表格信息"""
        try:
            # 获取表格的 HTML 内容
            table_result = region.get("tableResult", {})
            table_html = table_result.get("html", "")

            if not table_html:
                logger.warning("No HTML content in table region")
                return None

            # 解析 HTML 表格为结构化数据
            headers, rows = self._parse_html_table(table_html)

            if not headers or not rows:
                return None

            bbox = region.get("bbox", [])

            return {
                "id": f"page_{page_idx}_table_{int(bbox[0]) if bbox else 0}",
                "title": f"表格 (Page {page_idx + 1})",
                "headers": headers,
                "rows": rows,
                "rowCount": len(rows),
                "columnCount": len(headers),
                "confidence": 95.0,
                "html": table_html,
                "bbox": bbox,
                "page": page_idx
            }

        except Exception as e:
            logger.warning(f"Failed to extract table from region: {e}")
            return None

    def _extract_formula_from_region(
        self,
        region: Dict,
        page_idx: int
    ) -> Optional[Dict]:
        """从 region 中提取公式信息"""
        try:
            # 获取公式的 LaTeX 内容
            formula_result = region.get("formulaResult", {})
            latex = formula_result.get("latex", "")

            if not latex:
                # 尝试从 text 字段获取
                latex = region.get("text", "")

            if not latex:
                return None

            bbox = region.get("bbox", [])

            return {
                "id": f"page_{page_idx}_formula_{int(bbox[0]) if bbox else 0}",
                "latex": latex,
                "type": region.get("type", "formula"),
                "bbox": bbox,
                "page": page_idx,
                "confidence": 90.0
            }

        except Exception as e:
            logger.warning(f"Failed to extract formula from region: {e}")
            return None

    def _extract_tables_from_markdown(self, markdown_text: str, page_idx: int) -> list:
        """从 markdown 文本中提取 HTML 表格"""
        tables = []
        table_pattern = r'<table[^>]*>(.*?)</table>'
        matches = re.findall(table_pattern, markdown_text, re.DOTALL | re.IGNORECASE)

        logger.info(f"   Found {len(matches)} HTML tables in markdown")

        for table_idx, table_html in enumerate(matches):
            try:
                headers, rows = self._parse_html_table(f"<table>{table_html}</table>")

                if headers and rows:
                    table_info = {
                        "id": f"page_{page_idx}_table_{table_idx}",
                        "title": f"表格 {table_idx + 1} (Page {page_idx + 1})",
                        "headers": headers,
                        "rows": rows,
                        "rowCount": len(rows),
                        "columnCount": len(headers),
                        "confidence": 95.0,
                        "html": f"<table>{table_html}</table>",
                        "page": page_idx
                    }
                    tables.append(table_info)
            except Exception as e:
                logger.warning(f"Failed to parse table {table_idx} on page {page_idx}: {e}")

        return tables

    def _parse_html_table(self, table_html: str) -> tuple:
        """
        解析 HTML 表格

        Returns:
            (headers, rows) 元组
        """
        try:
            # 提取所有行
            row_pattern = r'<tr[^>]*>(.*?)</tr>'
            rows_html = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

            headers = []
            data_rows = []

            for row_idx, row_html in enumerate(rows_html):
                # 提取单元格（包括 th 和 td）
                cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
                cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

                # 清理 HTML 标签
                cells = [re.sub(r'<[^>]+>', '', cell).strip() for cell in cells]

                if row_idx == 0:
                    # 第一行作为表头
                    headers = cells
                else:
                    # 其他行作为数据
                    data_rows.append(cells)

            return headers, data_rows

        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            return [], []


# 全局服务实例
paddleocr_service = PaddleOCRService()
