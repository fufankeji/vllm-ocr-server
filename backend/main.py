#!/usr/bin/env python3
"""
OCR Analysis Backend Service
FastAPI server for PDF OCR processing with MinerU
"""

import os
import sys
import uvicorn
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.mineru_service import MinerUService
from app.services.deepseek_service import DeepSeekOCRService
from app.services.paddleocr_service import PaddleOCRService
from app.services.markdown_parser import MarkdownParser
from app.utils.file_utils import ensure_directories, cleanup_file
from app.models.ocr_models import OCRRequest, OCRResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Ensure directories exist
ensure_directories()

# Initialize FastAPI app
app = FastAPI(
    title="OCR Analysis Backend",
    description="Backend service for PDF OCR processing with MinerU",
    version="1.0.0"
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/exports", StaticFiles(directory="exports"), name="exports")

# 图片代理路由 - 支持 MinerU 和 DeepSeek-OCR API 服务器
@app.get("/images/{image_path:path}")
async def proxy_images(image_path: str):
    """代理图片请求到 MinerU 或 DeepSeek-OCR API 服务器"""
    import requests
    try:
        # 检查是否是DeepSeek图像路径
        if image_path.startswith("deepseek_img_"):
            # DeepSeek图像处理
            deepseek_host = os.getenv("DEEPSEEK_OCR_API_URL", "http://192.168.110.131:8797").replace("/ocr", "")
            # 提取图像ID
            image_id = image_path.replace("deepseek_img_", "")
            image_url = f"{deepseek_host}/images/{image_id}"
            logger.info(f"代理DeepSeek图片请求: {image_url}")
        else:
            # MinerU图像处理 (原逻辑)
            mineru_host = os.getenv("MINERU_API_URL", "http://192.168.110.131:50000").split("/file_parse")[0]
            image_url = f"{mineru_host}/images/{image_path}"
            logger.info(f"代理MinerU图片请求: {image_url}")

        response = requests.get(image_url, timeout=10)

        if response.status_code == 200:
            from fastapi.responses import Response
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "image/jpeg")
            )
        else:
            logger.warning(f"图片获取失败: {image_url}, 状态码: {response.status_code}")
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        logger.error(f"图片代理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch image: {str(e)}")

# Initialize services
mineru_service = MinerUService()
deepseek_service = DeepSeekOCRService()
paddleocr_service = PaddleOCRService()
markdown_parser = MarkdownParser()

# Global task storage for progress tracking
processing_tasks = {}

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "OCR Analysis Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    mineru_health = await mineru_service.check_health()

    return HealthResponse(
        status="healthy" if mineru_health.get("available") else "degraded",
        timestamp=str(Path().absolute()),
        services={
            "mineru": mineru_health
        }
    )

@app.post("/api/ocr/analyze", response_model=OCRResponse)
async def analyze_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("mineru"),
    options: str = Form("{}")
):
    """
    Analyze PDF file using specified OCR model

    Args:
        file: Uploaded PDF file
        model: OCR model to use ('mineru' or 'deepseek')
        options: JSON string of additional options

    Returns:
        OCR analysis results
    """

    # Debug logging for received parameters
    import json
    logger.info(f"🔍 Backend model debugging:")
    logger.info(f"  Received model parameter: '{model}'")
    logger.info(f"  Model type: {type(model)}")
    logger.info(f"  Model repr: {repr(model)}")
    logger.info(f"  Model stripped: '{model.strip() if model else model}'")

    supported_models = ["mineru", "deepseek", "paddleocr"]
    if model not in supported_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not supported. Available models: {', '.join(supported_models)}"
        )

    # Validate file type
    allowed_types = os.getenv("ALLOWED_FILE_TYPES", "application/pdf").split(",")
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file.content_type}' not allowed. Allowed types: {', '.join(allowed_types)}"
        )

    # Validate file size
    max_size = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size {file_size} exceeds maximum allowed size {max_size} bytes"
        )

    # Save uploaded file
    upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    file_path = upload_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File uploaded: {file.filename} ({file_size} bytes)")

        # Process based on selected model
        logger.info(f"Starting OCR analysis with {model}")

        if model == "deepseek":
            # Use DeepSeek OCR
            import json
            opts = json.loads(options) if options else {}
            result = await deepseek_service.analyze_document(file_path, opts)

            return OCRResponse(**result)

        elif model == "paddleocr":
            # Use PaddleOCR-VL
            logger.info("📘 Processing with PaddleOCR-VL...")
            result = await paddleocr_service.process_file(str(file_path))

            # 提取数据
            markdown_content = result.get("markdown", "")
            images = result.get("images", [])
            tables = result.get("tables", [])
            formulas_raw = result.get("formulas", [])

            # 转换公式格式
            formulas_formatted = []
            for formula in formulas_raw:
                formulas_formatted.append({
                    "id": formula.get("id", ""),
                    "type": formula.get("type", "formula"),
                    "formula": formula.get("latex", ""),
                    "description": f"Formula on page {formula.get('page', 0) + 1}",
                    "confidence": formula.get("confidence", 90.0),
                    "position": None
                })

            # 构建完整响应
            response_data = {
                "success": True,
                "model": "paddleocr",
                "filename": file.filename,
                "fullMarkdown": markdown_content,
                "results": {
                    "text": {
                        "fullText": markdown_content,
                        "textBlocks": [],
                        "keywords": [],
                        "confidence": 95.0,
                        "stats": {
                            "total_chars": len(markdown_content),
                            "total_pages": result.get("metadata", {}).get("total_pages", 0)
                        }
                    },
                    "tables": tables,
                    "formulas": formulas_formatted,
                    "images": images,
                    "handwritten": {
                        "detected": False,
                        "text": "No handwritten content detected",
                        "confidence": 0.0,
                        "areas": []
                    },
                    "performance": {
                        "accuracy": 95.0,
                        "speed": 0.0,
                        "memory": 0
                    },
                    "metadata": {
                        "totalElements": len(images) + len(tables) + len(formulas_formatted),
                        "contentTypes": ["text", "images", "tables", "formulas"],
                        "processingTime": None
                    }
                },
                "metadata": result.get("metadata", {})
            }

            return OCRResponse(**response_data)

        else:
            # Use MinerU (default)
            # Parse options
            import json
            opts = json.loads(options) if options else {}
            backend = opts.get('backend', os.getenv('MINERU_BACKEND', 'pipeline'))
            enable_ocr = opts.get('enable_ocr', True)
            language = opts.get('language', 'ch')
            device = opts.get('device', 'cuda:3')

            logger.info(f"🔧 MinerU options: backend={backend}, enable_ocr={enable_ocr}, language={language}, device={device}")

            # Parse PDF using MinerU
            parse_result = await mineru_service.parse_pdf(
                str(file_path),
                backend=backend,
                enable_ocr=enable_ocr,
                language=language,
                device=device
            )

            if not parse_result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"MinerU parsing failed: {parse_result.get('error', 'Unknown error')}"
                )

            # Parse markdown content
            markdown_file = parse_result.get("markdown_file")
            if not markdown_file or not Path(markdown_file).exists():
                raise HTTPException(
                    status_code=500,
                    detail="No markdown file generated by MinerU"
                )

            # Extract structured content from markdown using content_list data
            raw_data = parse_result.get("raw_data", {})

            # 调试：检查数据传递
            logger.info(f"🔍 parse_result keys: {parse_result.keys()}")
            logger.info(f"🔍 raw_data keys: {raw_data.keys()}")
            logger.info(f"🔍 images type: {type(raw_data.get('images'))}")
            if raw_data.get('images'):
                if isinstance(raw_data.get('images'), dict):
                    logger.info(f"🔍 images keys: {list(raw_data.get('images').keys())[:3]}")
                    logger.info(f"🔍 images sample key type: {type(list(raw_data.get('images').keys())[0]) if raw_data.get('images') else 'N/A'}")
                else:
                    logger.info(f"🔍 images is not dict, type: {type(raw_data.get('images'))}")
                    logger.info(f"🔍 images value preview: {str(raw_data.get('images'))[:200]}...")
            else:
                logger.warning("⚠️  raw_data中images为None或空")

            structured_content = await markdown_parser.parse_with_content_list(
                markdown_content=parse_result.get("content", ""),
                content_list=raw_data.get("content_list"),
                middle_json=raw_data.get("middle_json"),
                images_data=raw_data.get("images")
            )

            # Keep files for user access - don't cleanup
            logger.info(f"OCR analysis completed for {file.filename}")
            logger.info(f"PDF saved to: {file_path}")
            logger.info(f"Markdown saved to: {markdown_file}")

            # Read the complete markdown content
            with open(markdown_file, 'r', encoding='utf-8') as f:
                full_markdown = f.read()

            return OCRResponse(
                success=True,
                model=model,
                filename=file.filename,
                results=structured_content,
                fullMarkdown=full_markdown,
                metadata=parse_result.get("metadata", {})
            )

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        # Cleanup on error
        if file_path.exists():
            background_tasks.add_task(cleanup_file, file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

@app.get("/api/ocr/status/{task_id}")
async def get_task_status(task_id: str):
    """Get processing status for a task"""
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )

    return processing_tasks[task_id]

@app.get("/exports/{filename}")
async def download_file(filename: str):
    """Download exported file"""
    file_path = Path("exports") / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    return FileResponse(
        file_path,
        media_type='application/octet-stream',
        filename=filename
    )

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    logger.info(f"Starting OCR Analysis Backend on {host}:{port}")
    logger.info(f"MinerU MCP URL: {os.getenv('MCP_SERVER_URL', 'Not configured')}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )