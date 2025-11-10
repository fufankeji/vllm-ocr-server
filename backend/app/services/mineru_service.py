#!/usr/bin/env python3
"""
MinerU Service
"""

import asyncio
import json
import logging
import os
import requests
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# PDF和图像处理库
try:
    import pypdfium2
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("pypdfium2 or PIL not available - image extraction will be disabled")

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class MinerUService:
    """MinerU 服务类"""

    def __init__(self):
        # 使用 ocr_v2_extractors.py 中的配置
        self.api_url = os.getenv("MINERU_API_URL", "http://192.168.110.131:50000/file_parse")
        self.vllm_url = os.getenv("VLLM_SERVER_URL", "http://192.168.110.131:30000")
        self.backend = os.getenv("MINERU_BACKEND", "pipeline")
        self.timeout = int(os.getenv("MINERU_TIMEOUT", "600"))

        # 可视化输出目录
        self.viz_base_dir = Path(os.getenv(
            "MINERU_VIZ_DIR",
            "/home/MuyuWorkSpace/05_OcrProject/backend/mineru_visualizations"
        ))
        self.viz_base_dir.mkdir(parents=True, exist_ok=True)

    async def parse_pdf(
        self,
        pdf_path: str,
        backend: str = "pipeline",
        enable_ocr: bool = True,
        language: str = "ch",
        device: str = "cuda:3"
    ) -> Dict[str, Any]:
        """
        解析PDF文件 - 使用 ocr_v2_extractors.py 中的 MinerUExtractor 逻辑

        Args:
            pdf_path: PDF文件路径
            backend: 后端类型
            enable_ocr: 是否启用OCR
            language: 文档语言
            device: 设备

        Returns:
            解析结果字典，包含markdown和结构化数据
        """
        try:
            logger.info(f"[MinerU] 开始解析PDF: {Path(pdf_path).name}")

            # 检查文件是否存在
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                return {
                    "success": False,
                    "error": f"PDF文件不存在: {pdf_path}"
                }

            logger.info(f"文件: {pdf_file.name}")
            logger.info(f"大小: {pdf_file.stat().st_size / 1024:.2f} KB")

            # 调用 MinerU API
            logger.info(f"调用 MinerU API: {self.api_url}")

            # 1. 调用MinerU API
            with open(pdf_file, 'rb') as f:
                files = [('files', (pdf_file.name, f, 'application/pdf'))]
                data = {
                    'backend': backend,
                    'server_url': self.vllm_url,
                    'parse_method': 'auto',
                    'lang_list': language,
                    'return_md': 'true',
                    'return_middle_json': 'true',
                    'return_model_output': 'true',
                    'return_content_list': 'true',
                    'start_page_id': '0',
                    'end_page_id': '99999',
                }

                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                raise Exception(f"MinerU API返回错误: {response.status_code}")

            # 2. 解析返回结果（参考 ocr_v2_extractors.py:105-125）
            if response.headers.get("content-type", "").startswith("application/json"):
                file_json = response.json()
            else:
                file_json = json.loads(response.text)

            # 提取顶层信息
            backend = file_json.get("backend", self.backend)
            version = file_json.get("version", "2.5.4")

            # 提取结果数据（文件名作为key，去掉.pdf后缀）
            results = file_json.get("results", {})
            if not results:
                raise Exception("MinerU返回results为空")

            # 获取第一个结果（通常只有一个PDF文件）
            file_key = list(results.keys())[0] if results else None
            if not file_key:
                raise Exception("MinerU返回结果为空")

            res = results[file_key]

            # 解析各个部分（
            md_content = res.get("md_content", "")

            # 解析JSON字符串
            def safe_json_loads(text):
                if not isinstance(text, str):
                    return text
                try:
                    return json.loads(text.strip())
                except:
                    return None

            middle_json = safe_json_loads(res.get("middle_json"))
            model_output = safe_json_loads(res.get("model_output"))
            content_list = safe_json_loads(res.get("content_list"))

            # 调试 images 数据
            images_raw = res.get("images")
            logger.info(f"images_raw type: {type(images_raw)}")
            if images_raw:
                logger.info(f"images_raw preview: {str(images_raw)[:200]}...")
            else:
                logger.warning("API返回的images_raw为None或空 - 这是正常的,50000端口API不返回images")

            images = safe_json_loads(images_raw)
            logger.info(f"images after safe_json_loads type: {type(images)}")
            if images:
                if isinstance(images, dict):
                    logger.info(f"images keys: {list(images.keys())[:3]}")
                else:
                    logger.info(f"images is not dict: {type(images)}")
            else:
                # 50000端口API不返回images,需要从PDF中提取图片
                logger.info("从PDF提取图片数据")
                images = self._extract_images_from_pdf(pdf_file, content_list, middle_json)
                logger.info(f"提取了 {len(images)} 个图片")

            page_images = safe_json_loads(res.get("page_images"))

            # 统计信息
            total_pages = len(middle_json.get("pdf_info", [])) if middle_json else 0
            total_images = self._count_images(middle_json)

            logger.info("解析完成！")
            logger.info(f"   - Markdown 长度: {len(md_content)} 字符")
            logger.info(f"   - 总页数: {total_pages}")
            logger.info(f"   - 图片数量: {total_images}")
            logger.info(f"   - Content_list 条目: {len(content_list) if content_list else 0}")

            # 3. 保存结果到本地（保持与原来相同的逻辑）
            output_dir = Path(os.getenv("TEMP_DIR", "./temp"))
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{pdf_file.stem}_parsed.md"
            output_file.write_text(md_content, encoding='utf-8')

            logger.info(f"Markdown已保存到: {output_file.absolute()}")

            # 4. 返回完整结果（包含结构化数据）
            return {
                "success": True,
                "content": md_content,
                "markdown_file": str(output_file),
                "raw_data": {
                    # 保留原始 API 返回的所有字段（参考 ocr_v2_extractors.py:180-202）
                    'md_content': md_content,
                    'middle_json': middle_json,  # 已解析的对象（不是字符串）
                    'model_output': model_output,  # 已解析的对象
                    'content_list': content_list,  # 已解析的对象
                    'images': images,  # 图片数据
                    'page_images': page_images,  # 页面图片数据
                    # 添加顶层信息
                    'backend': backend,
                    'version': version,
                },
                "metadata": {
                    "backend": backend,
                    "version": version,
                    "total_pages": total_pages,
                    "total_images": total_images,
                    "content_list_count": len(content_list) if content_list else 0
                },
                "stats": {
                    "totalCharacters": len(md_content),
                    "totalLines": md_content.count(chr(10)) + 1,
                    "fileSizeKB": output_file.stat().st_size / 1024
                }
            }

        except Exception as e:
            logger.error(f"[MinerU] 解析PDF失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"MinerU解析失败: {str(e)}"
            }

    def _count_images(self, middle_json: dict) -> int:
        """统计图片数量（参考 ocr_v2_extractors.py:210-221）"""
        if not middle_json:
            return 0

        count = 0
        pdf_info = middle_json.get("pdf_info", [])
        for page in pdf_info:
            for block in page.get("preproc_blocks", []):
                if block.get("type") == "image":
                    count += 1
        return count

    def _calculate_transform_params(
        self,
        page_idx: int,
        middle_json: dict,
        content_list: list
    ) -> tuple:
        """计算坐标转换参数（来自参考代码）"""
        SCALE_X = SCALE_Y = 1.0
        OFFSET_X = OFFSET_Y = 0.0

        if not middle_json:
            return (SCALE_X, SCALE_Y, OFFSET_X, OFFSET_Y)

        pdf_info = middle_json.get("pdf_info", [])
        if page_idx >= len(pdf_info):
            return (SCALE_X, SCALE_Y, OFFSET_X, OFFSET_Y)

        middle_blocks = pdf_info[page_idx].get("preproc_blocks", [])
        content_items = [
            item for item in content_list
            if isinstance(item, dict) and item.get("page_idx") == page_idx
        ]

        if middle_blocks and content_items:
            m_bbox = middle_blocks[0].get("bbox", [])
            c_bbox = content_items[0].get("bbox", [])

            if len(m_bbox) >= 4 and len(c_bbox) >= 4:
                SCALE_X = (c_bbox[2] - c_bbox[0]) / (m_bbox[2] - m_bbox[0]) if (m_bbox[2] - m_bbox[0]) > 0 else 1.0
                SCALE_Y = (c_bbox[3] - c_bbox[1]) / (m_bbox[3] - m_bbox[1]) if (m_bbox[3] - m_bbox[1]) > 0 else 1.0
                OFFSET_X = c_bbox[0] - m_bbox[0] * SCALE_X
                OFFSET_Y = c_bbox[1] - m_bbox[1] * SCALE_Y

        return (SCALE_X, SCALE_Y, OFFSET_X, OFFSET_Y)

    def _transform_bbox(
        self,
        bbox: list,
        page_idx: int,
        middle_json: dict,
        content_list: list,
        W_img: int,
        H_img: int,
        W_pdf: float,
        H_pdf: float
    ) -> list:
        """转换 bbox 坐标（来自参考代码）"""
        # 获取转换参数
        SCALE_X, SCALE_Y, OFFSET_X, OFFSET_Y = self._calculate_transform_params(
            page_idx, middle_json, content_list
        )

        # 1. 去除偏移
        bbox_no_offset = [
            bbox[0] - OFFSET_X,
            bbox[1] - OFFSET_Y,
            bbox[2] - OFFSET_X,
            bbox[3] - OFFSET_Y
        ]

        # 2. 反缩放到 PDF 坐标系
        bbox_in_pdf = [
            bbox_no_offset[0] / SCALE_X if SCALE_X != 0 else bbox_no_offset[0],
            bbox_no_offset[1] / SCALE_Y if SCALE_Y != 0 else bbox_no_offset[1],
            bbox_no_offset[2] / SCALE_X if SCALE_X != 0 else bbox_no_offset[2],
            bbox_no_offset[3] / SCALE_Y if SCALE_Y != 0 else bbox_no_offset[3]
        ]

        # 3. 缩放到图片坐标系
        sx = W_img / W_pdf if W_pdf > 0 else 1.0
        sy = H_img / H_pdf if H_pdf > 0 else 1.0
        final_bbox = [
            bbox_in_pdf[0] * sx,
            bbox_in_pdf[1] * sy,
            bbox_in_pdf[2] * sx,
            bbox_in_pdf[3] * sy
        ]

        return final_bbox

    def _extract_images_from_pdf(
        self,
        pdf_path: Path,
        content_list: list,
        middle_json: dict
    ) -> dict:
        """从PDF中提取图片并编码为base64"""
        if not PIL_AVAILABLE or not content_list:
            logger.warning("PIL or pypdfium2 not available, cannot extract images")
            return {}

        try:
            images_dict = {}

            # 打开PDF文档
            doc = pypdfium2.PdfDocument(str(pdf_path))
            logger.info(f"📖 打开PDF文档，共 {len(doc)} 页")

            # 创建页面图片缓存
            page_images_cache = {}

            # 遍历content_list中的图片
            for item in content_list:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")
                if item_type != "image":
                    continue

                img_path = item.get("img_path", "")
                bbox = item.get("bbox", [])
                page_idx = item.get("page_idx", 0)

                if not img_path or len(bbox) < 4:
                    continue

                try:
                    # 渲染页面图片（如果还没渲染）
                    if page_idx not in page_images_cache:
                        if page_idx >= len(doc):
                            logger.warning(f"⚠️  页面索引 {page_idx} 超出范围")
                            continue

                        page = doc.get_page(page_idx)
                        pil_img = page.render(scale=2.0).to_pil()  # 使用2.0缩放以获得更好质量
                        page_images_cache[page_idx] = pil_img
                        logger.info(f"📄 渲染页面 {page_idx}, 尺寸: {pil_img.size}")

                    page_img = page_images_cache[page_idx]
                    W_img, H_img = page_img.size

                    # 获取PDF页面尺寸
                    pdf_info = middle_json.get("pdf_info", [])
                    if page_idx < len(pdf_info):
                        page_info = pdf_info[page_idx]
                        W_pdf = page_info.get("width", 595)
                        H_pdf = page_info.get("height", 841)
                    else:
                        W_pdf, H_pdf = 595, 841

                    # 使用参考代码的坐标转换逻辑
                    final_bbox = self._transform_bbox(
                        bbox, page_idx, middle_json, content_list, W_img, H_img, W_pdf, H_pdf
                    )

                    # 转换为整数坐标并裁切
                    x1 = max(0, int(round(final_bbox[0])))
                    y1 = max(0, int(round(final_bbox[1])))
                    x2 = min(W_img, int(round(final_bbox[2])))
                    y2 = min(H_img, int(round(final_bbox[3])))

                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"⚠️  无效的bbox: {bbox}")
                        continue

                    # 裁剪图片
                    cropped = page_img.crop((x1, y1, x2, y2))

                    # 自动裁剪白边
                    try:
                        # 转换为RGB模式以便处理
                        if cropped.mode != 'RGB':
                            cropped = cropped.convert('RGB')

                        # 使用getbbox()自动检测非白色区域
                        # 先转换为灰度图，然后反转（让白色变成黑色）
                        import numpy as np
                        img_array = np.array(cropped)

                        # 计算每个像素的亮度
                        gray = np.mean(img_array, axis=2)

                        # 找到非白色区域（阈值240，接近白色的也算白色）
                        mask = gray < 240

                        # 获取非白色区域的边界
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)

                        if rows.any() and cols.any():
                            rmin, rmax = np.where(rows)[0][[0, -1]]
                            cmin, cmax = np.where(cols)[0][[0, -1]]

                            # 裁剪掉白边，保留2像素的边距
                            margin = 2
                            rmin = max(0, rmin - margin)
                            cmin = max(0, cmin - margin)
                            rmax = min(cropped.height - 1, rmax + margin)
                            cmax = min(cropped.width - 1, cmax + margin)

                            cropped = cropped.crop((cmin, rmin, cmax + 1, rmax + 1))
                    except Exception as e:
                        logger.warning(f"⚠️  自动裁剪白边失败: {str(e)}")
                        # 如果失败，继续使用原始裁剪的图片

                    # 转换为base64
                    buffered = BytesIO()
                    cropped.save(buffered, format="PNG")
                    img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

                    images_dict[img_path] = img_base64
                    logger.info(f"✅ 提取图片: {img_path}, 尺寸: {x2-x1}x{y2-y1}")

                except Exception as e:
                    logger.error(f"❌ 提取图片失败 {img_path}: {str(e)}")

            doc.close()
            return images_dict

        except Exception as e:
            logger.error(f"❌ PDF图片提取失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    async def check_health(self) -> Dict[str, Any]:
        """检查MinerU API服务是否可用"""
        try:
            response = requests.get(f"{self.api_url.replace('/file_parse', '/health')}", timeout=5)
            if response.status_code == 200:
                return {
                    "available": True,
                    "url": self.api_url,
                    "status": response.status_code,
                    "response": response.json()
                }
            else:
                return {
                    "available": False,
                    "url": self.api_url,
                    "status": response.status_code,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "available": False,
                "url": self.api_url,
                "error": str(e)
            }

    async def get_available_tools(self) -> Dict[str, Any]:
        """获取MinerU API提供的功能信息"""
        try:
            logger.info(f"🔌 检查 MinerU API: {self.api_url}")

            # 尝试简单的健康检查
            response = requests.get(f"{self.api_url.replace('/file_parse', '/health')}", timeout=5)

            if response.status_code == 200:
                logger.info("✅ MinerU API 服务可用")
                return {
                    "success": True,
                    "tools": [
                        {
                            "name": "file_parse",
                            "description": "MinerU文档解析API",
                            "parameters": {
                                "file": "PDF文件",
                                "backend": "处理后端类型",
                                "return_md": "返回markdown",
                                "return_middle_json": "返回中间JSON",
                                "return_content_list": "返回内容列表"
                            }
                        }
                    ],
                    "count": 1
                }
            else:
                logger.warning(f"⚠️ MinerU API 响应异常: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API响应异常: HTTP {response.status_code}"
                }

        except Exception as e:
            logger.error(f"❌ MinerU API 连接失败: {str(e)}")
            return {
                "success": False,
                "error": f"连接MinerU API失败: {str(e)}"
            }