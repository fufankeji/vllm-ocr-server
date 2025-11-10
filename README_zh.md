<div align="center">
  <h1>LangChain1.0 + OCR 多模态文档解析系统</h1>
  <p><em>集成 MinerU、PaddleOCR-VL、DeepSeekOCR 三大行业内性能最强的OCR解析项目</em></p>
  <span>中文 | <a href="./README.md">English</a></span>
</div>

## ⚡ 项目简介
通过vLLM推理框架部署目前行业内性能最强的OCR解析项目：MinerU、DeepSeek-OCR和PaddleOCR-VL，并实现具有统一解析服务接口的多模态数据分析系统。包含针对`DeepSeek-OCR`、`MinerU`等服务接口的优化及封装，可以直接在企业中落地。


https://github.com/user-attachments/assets/d2ef05be-fa3b-4037-9ce5-49be52cc71b5



其中 MinerU、PaddleOCR-VL、DeepSeekOCR 安装和详解请参考 <a href="./Deployment.md">教程</a>

## 🎯 主要功能

 - 统一解析接口：MinerU、PaddleOCR‑VL、DeepSeek‑OCR 可插拔选择
 - 批量解析：支持批量处理 PDF 与图片，多页文档自动拆分
 - 高性能部署：基于 vLLM 推理框架
 - 多模态支持：文本、表格、公式、图片等多模态内容抽取
 - 标准化输出：统一格式返回，支持 Markdown/JSON 及图片导出




## 🚀 快速开始

其中 MinerU、PaddleOCR-VL、DeepSeekOCR 安装和详解请参考 <a href="./Deployment.md">教程</a>

### 后端服务启动
配置 `backend/.env`文件
```
# Server Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=True

# MinerU Configuration - Using Direct API
MINERU_API_URL=http://192.168.130.4:50000/file_parse
VLLM_SERVER_URL=http://192.168.130.4:40000
MINERU_BACKEND=vlm-vllm-async-engine
MINERU_TIMEOUT=600
MINERU_VIZ_DIR=/home/MuyuWorkSpace/05_OcrProject/backend/mineru_visualizations

# DeepSeek OCR Configuration
DEEPSEEK_OCR_API_URL=http://192.168.130.4:8797/ocr

# PaddleOCR Configuration
PADDLEOCR_API_URL=http://192.168.130.4:10800/layout-parsing

# File Upload Limits
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=application/pdf,image/png,image/jpeg,image/jpg,image/webp

# Storage Paths
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
TEMP_DIR=./temp

# Processing Timeout (seconds)
OCR_TIMEOUT=300

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```
启动后端服务
```bash
    cd backend

    # 创建并激活虚拟环境
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # 或
    venv\Scripts\activate     # Windows

    # 安装依赖
    pip install -r requirements.txt

    # 启动服务
    python main.py
```

### 前端服务启动
```bash
    cd frontend

    # 安装依赖
    npm install

    # 启动开发服务器
    npm run dev
```

## 🙈 贡献
欢迎通过GitHub提交 PR 或者issues来对项目进行贡献。我们非常欢迎任何形式的贡献，包括功能改进、bug修复或是文档优化。

## 😎 技术交流
探索我们的技术社区 👉 [大模型技术社区丨赋范空间](https://kq4b3vgg5b.feishu.cn/wiki/JuJSwfbwmiwvbqkiQ7LcN1N1nhd)

扫描添加小可爱，回复“OCR”加入技术交流群，与其他小伙伴一起交流学习。
<div align="center">
<img src="assets\交流群.jpg" width="200" alt="技术交流群二维码">
<div>
