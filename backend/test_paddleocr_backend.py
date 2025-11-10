#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试后端 PaddleOCR 集成
通过后端 API 测试 PaddleOCR 功能
"""

import requests
import sys
from pathlib import Path


def test_paddleocr_backend(pdf_path: str):
    """
    测试后端 PaddleOCR API

    Args:
        pdf_path: PDF 文件路径
    """
    backend_url = "http://localhost:8000/api/ocr/analyze"

    print(f"📄 测试文件: {pdf_path}")
    print(f"🌐 后端地址: {backend_url}")
    print("-" * 60)

    try:
        # 读取文件
        with open(pdf_path, "rb") as f:
            files = {"file": (Path(pdf_path).name, f, "application/pdf")}
            data = {"model": "paddleocr"}

            print("🚀 发送请求到后端...")
            response = requests.post(backend_url, files=files, data=data, timeout=300)

        if response.status_code != 200:
            print(f"❌ 后端返回错误: {response.status_code}")
            print(f"错误信息: {response.text[:500]}")
            return False

        # 解析响应
        result = response.json()

        print("✅ 后端响应成功")
        print(f"📋 响应字段: {list(result.keys())}")
        print("-" * 60)

        # 提取结果
        success = result.get("success", False)
        message = result.get("message", "")
        results = result.get("results", {})
        metadata = result.get("metadata", {})

        print(f"✅ 处理状态: {'成功' if success else '失败'}")
        print(f"📝 消息: {message}")
        print(f"📄 总页数: {metadata.get('total_pages', 0)}")
        print("-" * 60)

        # 统计结果
        markdown = results.get("markdown", "")
        images = results.get("images", [])
        tables = results.get("tables", [])
        formulas = results.get("formulas", [])

        print("📊 提取结果统计:")
        print(f"   - Markdown 长度: {len(markdown)} 字符")
        print(f"   - 图片数量: {len(images)}")
        print(f"   - 表格数量: {len(tables)}")
        print(f"   - 公式数量: {len(formulas)}")

        # 显示详细信息
        if images:
            print("\n🖼️  图片列表:")
            for idx, img in enumerate(images[:5]):  # 只显示前 5 个
                print(f"   {idx + 1}. {img.get('id')} - {img.get('description')}")
                print(f"      类型: {img.get('type')}, 页码: {img.get('page')}")

        if tables:
            print("\n📋 表格列表:")
            for idx, table in enumerate(tables[:3]):  # 只显示前 3 个
                print(f"   {idx + 1}. {table.get('title')}")
                print(f"      行数: {table.get('rowCount')}, 列数: {table.get('columnCount')}")
                print(f"      页码: {table.get('page')}")

        if formulas:
            print("\n🔢 公式列表:")
            for idx, formula in enumerate(formulas[:5]):
                print(f"   {idx + 1}. {formula.get('id')} - {formula.get('type')}")
                print(f"      LaTeX: {formula.get('latex', '')[:50]}...")

        print("\n" + "-" * 60)
        print("🎉 测试完成！")

        return True

    except FileNotFoundError:
        print(f"❌ 文件不存在: {pdf_path}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到后端服务")
        print("💡 请确保后端服务正在运行: python main.py")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_paddleocr_backend.py <pdf_path>")
        print("示例: python test_paddleocr_backend.py ../阿里开发手册-泰山版-2页.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    success = test_paddleocr_backend(pdf_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
