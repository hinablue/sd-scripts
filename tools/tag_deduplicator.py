#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
標籤去重處理工具
功能：讀取資料夾中的 txt 檔案，去除每行中重複的標籤，並輸出到新檔案
"""

import os
import glob
import argparse
from pathlib import Path


def process_tags_line(line):
    """
    處理單行標籤資料

    Args:
        line (str): 包含逗號分隔標籤的字串

    Returns:
        str: 去重後重組的標籤字串
    """
    # 移除行首行尾空白
    line = line.strip()

    # 如果是空行，直接返回
    if not line:
        return ""

    # 以逗號分割標籤
    tags = [tag.strip() for tag in line.split(',')]

    # 去除空標籤
    tags = [tag for tag in tags if tag]

    # 去除重複標籤，保持原始順序
    unique_tags = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            unique_tags.append(tag)
            seen.add(tag)

    # 重組為逗號分隔的字串
    return ', '.join(unique_tags)


def process_txt_file(input_file_path, output_file_path):
    """
    處理單個 txt 檔案

    Args:
        input_file_path (str): 輸入檔案路徑
        output_file_path (str): 輸出檔案路徑
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()

        processed_lines = []
        for line in lines:
            processed_line = process_tags_line(line)
            # 只保留非空行
            if processed_line:
                processed_lines.append(processed_line)

        # 寫入處理後的內容到新檔案
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in processed_lines:
                output_file.write(line + '\n')

        print(f"✅ 已處理: {input_file_path} -> {output_file_path}")
        print(f"   原始行數: {len(lines)}, 處理後行數: {len(processed_lines)}")

    except Exception as e:
        print(f"❌ 處理檔案時發生錯誤 {input_file_path}: {e}")


def parse_arguments():
    """
    解析命令列參數

    Returns:
        argparse.Namespace: 解析後的參數
    """
    parser = argparse.ArgumentParser(
        description='標籤去重處理工具 - 處理 txt 檔案中重複的標籤',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python tag_deduplicator.py -i input_folder -o output_folder
  python tag_deduplicator.py --input ./data --output ./results
  python tag_deduplicator.py -i /path/to/input -o /path/to/output
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='input_txt_files',
        help='輸入資料夾路徑 (預設: input_txt_files)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output_txt_files',
        help='輸出資料夾路徑 (預設: output_txt_files)'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='_processed',
        help='輸出檔案名稱後綴 (預設: _processed)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='標籤去重處理工具 v1.1'
    )

    return parser.parse_args()


def main():
    """
    主函數：處理資料夾中的所有 txt 檔案
    """
    # 解析命令列參數
    args = parse_arguments()

    input_folder = args.input
    output_folder = args.output
    suffix = args.suffix

    print(f"🔧 設定資訊:")
    print(f"   輸入資料夾: {input_folder}")
    print(f"   輸出資料夾: {output_folder}")
    print(f"   檔案後綴: {suffix}")
    print("-" * 50)

    # 檢查輸入資料夾是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 輸入資料夾不存在: {input_folder}")
        print(f"請創建 '{input_folder}' 資料夾並放入要處理的 txt 檔案")
        print(f"或使用 -i 參數指定其他輸入資料夾")
        return

    # 創建輸出資料夾（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 輸出資料夾已準備: {output_folder}")

    # 尋找所有 txt 檔案
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))

    if not txt_files:
        print(f"❌ 在 '{input_folder}' 資料夾中找不到任何 txt 檔案")
        return

    print(f"🔍 找到 {len(txt_files)} 個 txt 檔案")
    print("開始處理...")
    print("-" * 50)

    # 處理每個 txt 檔案
    for input_file_path in txt_files:
        # 取得檔案名稱（不含路徑）
        filename = os.path.basename(input_file_path)

        # 產生輸出檔案名稱（加上指定後綴）
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}{suffix}.txt"
        output_file_path = os.path.join(output_folder, output_filename)

        # 處理檔案
        process_txt_file(input_file_path, output_file_path)

    print("-" * 50)
    print(f"✅ 所有檔案處理完成！輸出檔案位於: {output_folder}")


if __name__ == "__main__":
    main()