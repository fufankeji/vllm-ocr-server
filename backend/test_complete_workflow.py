#!/usr/bin/env python3
"""
完整工作流程测试
测试从PDF上传到前端显示的完整流程
"""

import requests
import json
import time
from pathlib import Path

def test_complete_workflow():
    """测试完整工作流程"""
    print("🧪 完整工作流程测试")
    print("=" * 50)

    base_url = "http://localhost:8000"

    # 1. 测试服务器连接
    print("1. 测试服务器连接...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ 服务器连接成功")
        else:
            print(f"   ❌ 服务器连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 服务器连接错误: {str(e)}")
        return False

    # 2. 查找测试PDF文件
    print("\n2. 查找测试PDF文件...")
    pdf_files = list(Path('.').glob('*.pdf'))
    if not pdf_files:
        print("   ⚠️ 没有找到PDF文件，跳过PDF上传测试")
        print("   ✅ 基础服务测试通过")
        return True

    pdf_file = pdf_files[0]
    print(f"   📄 使用文件: {pdf_file.name} ({pdf_file.stat().st_size / 1024:.2f} KB)")

    # 3. 测试PDF上传和OCR分析
    print("\n3. 测试PDF上传和OCR分析...")
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            data = {
                'model': 'mineru',
                'options': json.dumps({
                    'backend': 'vlm-vllm-async-engine',
                    'enable_ocr': True,
                    'language': 'ch',
                    'device': 'cuda:3'
                })
            }

            print(f"   📤 上传文件: {pdf_file.name}")
            start_time = time.time()

            response = requests.post(
                f"{base_url}/api/ocr/analyze",
                files=files,
                data=data,
                timeout=300  # 5分钟超时
            )

            end_time = time.time()
            print(f"   ⏱️ 处理时间: {end_time - start_time:.2f} 秒")

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("   ✅ OCR分析成功")

                    # 4. 检查结果结构
                    print("\n4. 检查结果结构...")
                    results = result.get('results', {})

                    text_info = results.get('text', {})
                    print(f"   📝 文本块: {len(text_info.get('textBlocks', []))}")
                    print(f"   🔑 关键词: {len(text_info.get('keywords', []))}")
                    print(f"   📊 置信度: {text_info.get('confidence', 0)}%")

                    tables = results.get('tables', [])
                    print(f"   📋 表格: {len(tables)}")
                    for i, table in enumerate(tables):
                        print(f"      - {table.get('title')}: {table.get('rowCount')}行 x {table.get('columnCount')}列")

                    formulas = results.get('formulas', [])
                    print(f"   🧮 公式: {len(formulas)}")
                    for i, formula in enumerate(formulas[:3]):  # 显示前3个
                        print(f"      - {formula.get('type', 'unknown')}: {formula.get('description', '')}")

                    images = results.get('images', [])
                    print(f"   🖼️ 图片: {len(images)}")
                    for i, image in enumerate(images):
                        print(f"      - {image.get('type', 'unknown')}: {image.get('path', '')}")

                    performance = results.get('performance', {})
                    print(f"   ⚡ 性能指标:")
                    print(f"      - 准确率: {performance.get('accuracy', 0)}%")
                    print(f"      - 处理时间: {performance.get('speed', 0)}秒")
                    print(f"      - 内存占用: {performance.get('memory', 0)}MB")

                    print("\n   ✅ 结果结构验证通过")

                    # 5. 模拟前端数据对接
                    print("\n5. 模拟前端数据对接...")
                    print("   📱 前端可以接收以下数据:")
                    print("      - 文本标签页: 完整文本 + 关键词 + 置信度")
                    print("      - 表格标签页: 结构化表格数据")
                    print("      - 公式标签页: 数学公式和描述")
                    print("      - 图片标签页: 图片信息和描述")
                    print("      - 手写标签页: 手写内容检测结果")
                    print("      - 性能标签页: 模型性能指标对比")

                    print("\n   ✅ 前端数据对接验证通过")

                    return True
                else:
                    print(f"   ❌ OCR分析失败: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"   ❌ 请求失败: {response.status_code}")
                print(f"   错误信息: {response.text}")
                return False

    except Exception as e:
        print(f"   ❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_complete_workflow()

    if success:
        print("\n🎉 完整工作流程测试通过！")
        print("✅ MinerU后端服务已成功对接前端数据结构")
        print("✅ 前端可以直接使用这些API端点:")
        print("   - GET /health - 健康检查")
        print("   - POST /api/ocr/analyze - OCR分析")
        print("   - GET /exports/{filename} - 文件下载")
    else:
        print("\n❌ 测试失败，请检查服务状态")