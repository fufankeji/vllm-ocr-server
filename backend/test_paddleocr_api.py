#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 PaddleOCR API
直接调用 API 并保存返回的 markdown 和结构化数据
"""

import base64
import json
import requests
import sys
from pathlib import Path


def test_paddleocr_api(file_path: str, output_dir: str = "./test_output"):
    """
    测试 PaddleOCR API

    Args:
        file_path: PDF 或图片文件路径
        output_dir: 输出目录
    """
    # API 配置
    api_url = "http://192.168.110.131:10800/layout-parsing"

    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📄 测试文件: {file_path}")
    print(f"🌐 API 地址: {api_url}")
    print(f"📂 输出目录: {output_dir}")
    print("-" * 60)

    try:
        # 读取文件并转为 Base64
        print("📤 读取文件并编码为 base64...")
        with open(file_path, "rb") as f:
            file_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 判断文件类型
        file_extension = Path(file_path).suffix.lower()
        if file_extension == ".pdf":
            file_type = 0  # PDF
        elif file_extension in [".png", ".jpg", ".jpeg", ".bmp"]:
            file_type = 1  # 图片
        else:
            print(f"⚠️  未知文件类型: {file_extension}，默认按图片处理")
            file_type = 1

        # 构造 JSON 请求体
        payload = {
            "file": file_base64,
            "fileType": file_type,
            "prettifyMarkdown": True,
            "visualize": False,  # 不生成可视化图像
        }

        headers = {"Content-Type": "application/json"}

        print("🚀 发送请求到 PaddleOCR API...")

        # 发送请求
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(payload),
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

        # 保存完整响应用于分析
        full_response_file = output_path / f"{Path(file_path).stem}_paddleocr_full_response.json"
        with open(full_response_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 完整响应已保存: {full_response_file}")

        # 检查错误码
        error_code = result.get("errorCode")
        if error_code != 0:
            error_msg = result.get("errorMsg", "未知错误")
            print(f"❌ 服务端错误 (code: {error_code}): {error_msg}")
            return False

        # 提取结果
        result_data = result.get("result", {})
        layout_parsing_results = result_data.get("layoutParsingResults", [])

        print(f"📄 页数: {len(layout_parsing_results)}")
        print("-" * 60)

        # 合并所有页面的 markdown
        all_markdown = []
        all_stats = {
            "total_pages": len(layout_parsing_results),
            "total_blocks": 0,
            "images": 0,
            "tables": 0,
            "formulas": 0,
            "text": 0
        }

        for page_idx, page_result in enumerate(layout_parsing_results):
            print(f"\n📄 分析第 {page_idx + 1} 页...")

            # 获取 markdown
            markdown_data = page_result.get("markdown", {})
            markdown_text = markdown_data.get("text", "")

            all_markdown.append(f"\n\n# Page {page_idx + 1}\n\n")
            all_markdown.append(markdown_text)

            print(f"   - Markdown 长度: {len(markdown_text)} 字符")

            # 获取结构化数据
            layout_result = page_result.get("layoutResult", {})
            regions = layout_result.get("regions", [])

            all_stats["total_blocks"] += len(regions)

            # 统计各类型块
            block_types = {}
            for region in regions:
                region_type = region.get("type", "unknown")
                block_types[region_type] = block_types.get(region_type, 0) + 1

            print(f"   - 总块数: {len(regions)}")
            print(f"   - 块类型分布: {block_types}")

            # 更新统计
            all_stats["images"] += block_types.get("figure", 0) + block_types.get("image", 0)
            all_stats["tables"] += block_types.get("table", 0)
            all_stats["formulas"] += block_types.get("equation", 0)
            all_stats["text"] += block_types.get("text", 0) + block_types.get("title", 0)

        # 保存合并的 Markdown
        final_markdown = "".join(all_markdown)
        md_file = output_path / f"{Path(file_path).stem}_paddleocr.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(final_markdown)
        print(f"\n✅ Markdown 已保存: {md_file}")

        # 保存统计摘要
        summary_file = output_path / f"{Path(file_path).stem}_paddleocr_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        print(f"✅ 统计摘要已保存: {summary_file}")

        # 打印总体统计
        print("\n" + "-" * 60)
        print("📊 总体统计信息:")
        print(f"   - 总页数: {all_stats['total_pages']}")
        print(f"   - 总块数: {all_stats['total_blocks']}")
        print(f"   - 图像: {all_stats['images']} 个")
        print(f"   - 表格: {all_stats['tables']} 个")
        print(f"   - 公式: {all_stats['formulas']} 个")
        print(f"   - 文本块: {all_stats['text']} 个")
        print(f"   - Markdown 总长度: {len(final_markdown)} 字符")

        print("\n" + "-" * 60)
        print("🎉 测试完成！")

        return True

    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
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
        print("用法: python test_paddleocr_api.py <file_path> [output_dir]")
        print("示例: python test_paddleocr_api.py course.pdf ./test_output")
        print("     python test_paddleocr_api.py image.png ./test_output")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_output"

    success = test_paddleocr_api(file_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
