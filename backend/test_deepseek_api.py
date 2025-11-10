#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 DeepSeek-OCR API
直接调用 API 并保存返回的 markdown 和图像数据
"""

import requests
import json
import sys
from pathlib import Path


def test_deepseek_ocr(pdf_path: str, output_dir: str = "./test_output"):
    """
    测试 DeepSeek-OCR API

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
    """
    # API 配置
    api_url = "http://192.168.110.131:8797/ocr"

    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📄 测试文件: {pdf_path}")
    print(f"🌐 API 地址: {api_url}")
    print(f"📂 输出目录: {output_dir}")
    print("-" * 60)

    # 读取 PDF 文件
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (Path(pdf_path).name, f, 'application/pdf')}

            # API 参数
            data = {
                'enable_description': 'false',  # 是否生成图片描述
            }

            print("🚀 发送请求到 DeepSeek API...")

            # 发送请求
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=300
            )

            if response.status_code != 200:
                print(f"❌ API 返回错误: {response.status_code}")
                print(f"错误信息: {response.text[:500]}")
                return False

            # 解析响应
            result = response.json()

            print(f"✅ API 响应成功")
            print(f"📋 响应包含的字段: {list(result.keys())}")
            print("-" * 60)

            # 提取数据
            markdown_content = result.get("markdown", "")
            page_count = result.get("page_count", 0)
            images_data = result.get("images", {})

            print(f"📝 Markdown 长度: {len(markdown_content)} 字符")
            print(f"📄 页数: {page_count}")
            print(f"🖼️  图像数量: {len(images_data)}")

            if images_data:
                print(f"🖼️  图像列表:")
                for img_key in list(images_data.keys())[:10]:
                    img_size = len(images_data[img_key])
                    print(f"   - {img_key}: {img_size} 字符 (base64)")

            print("-" * 60)

            # 保存 Markdown
            md_file = output_path / f"{Path(pdf_path).stem}_deepseek.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✅ Markdown 已保存: {md_file}")

            # 保存完整响应
            json_file = output_path / f"{Path(pdf_path).stem}_response.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                # 为了避免文件过大，只保存图像的部分信息
                simplified_result = {
                    "markdown": markdown_content,
                    "page_count": page_count,
                    "images_count": len(images_data),
                    "image_keys": list(images_data.keys())
                }
                json.dump(simplified_result, f, ensure_ascii=False, indent=2)
            print(f"✅ 响应摘要已保存: {json_file}")

            # 保存图像数据（可选，如果需要）
            if images_data:
                images_file = output_path / f"{Path(pdf_path).stem}_images.json"
                with open(images_file, 'w', encoding='utf-8') as f:
                    json.dump(images_data, f, ensure_ascii=False, indent=2)
                print(f"✅ 图像数据已保存: {images_file}")

            # 统计信息
            print("-" * 60)
            print("📊 统计信息:")

            # 统计表格数量
            import re
            table_count = len(re.findall(r'<table>', markdown_content, re.IGNORECASE))
            print(f"   - HTML 表格: {table_count} 个")

            # 统计图片引用
            img_ref_count = len(re.findall(r'!\[.*?\]\(.*?\)', markdown_content))
            print(f"   - Markdown 图片引用: {img_ref_count} 个")

            # 统计行数
            line_count = len(markdown_content.split('\n'))
            print(f"   - Markdown 行数: {line_count} 行")

            print("-" * 60)
            print("🎉 测试完成！")

            return True

    except FileNotFoundError:
        print(f"❌ 文件不存在: {pdf_path}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python test_deepseek_api.py <pdf_path> [output_dir]")
        print("示例: python test_deepseek_api.py course.pdf ./test_output")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"

    success = test_deepseek_ocr(pdf_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
