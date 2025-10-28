import os
import argparse


def analyze_folder_per_file(folder_path, output_filename="total_result.txt"):
    """
    遍历一个文件夹中的所有.jsonl文件，为每个文件独立计算
    包含 '"judge": true' 的行数占比，并将结果写入一个文本文件。

    Args:
        folder_path (str): 需要分析的文件夹路径。
        output_filename (str): 保存结果的文件名。
    """
    # 检查提供的路径是否是一个有效的文件夹
    if not os.path.isdir(folder_path):
        print(f"错误：路径 '{folder_path}' 不是一个有效的文件夹。")
        return

    results = []
    print(f"开始分析文件夹: {os.path.abspath(folder_path)}")

    # 遍历文件夹中的所有文件，使用sorted()确保输出顺序一致
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            print(f"正在处理文件: {filename}...")

            total_lines = 0
            judge_true_lines = 0

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 忽略空行
                        if not line.strip():
                            continue
                        
                        total_lines += 1
                        # 直接检查字符串比解析JSON更高效，且能避免因格式错误中断
                        if '"judge": true' in line:
                            judge_true_lines += 1
            except Exception as e:
                print(f"  无法读取文件 {filename}。错误: {e}")
                continue  # 跳过此文件，继续处理下一个

            # 计算当前文件的占比
            if total_lines > 0:
                percentage = (judge_true_lines / total_lines) * 100
            else:
                percentage = 0.0

            # 将格式化的结果添加到列表中
            results.append(f"{filename}: {percentage:.2f}%\n")

    if not results:
        print("在指定文件夹中未找到 .jsonl 文件。")
        return

    # 将所有结果一次性写入输出文件
    try:
        # 将结果文件保存在当前运行脚本的目录下
        output_path = os.path.join(folder_path, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(results)
        
        print("\n分析完成。")
        print(f"结果已保存至: {output_path}")

    except Exception as e:
        print(f"\n写入输出文件时发生错误: {e}")


def main():
    """
    主函数，用于解析命令行参数并启动分析。
    """
    # 初始化参数解析器
    parser = argparse.ArgumentParser(
        description='统计一个文件夹内所有jsonl文件中 "judge": true 的行数占比。',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 添加一个必须的位置参数 'folder_path'
    parser.add_argument(
        '--folder_path', 
        "-f", 
        type=str, 
        help='包含 .jsonl 文件的目标文件夹路径。'
    )
    
    # 解析命令行传入的参数
    args = parser.parse_args()
    
    # 使用传入的文件夹路径调用分析函数
    analyze_folder_per_file(args.folder_path)

if __name__ == "__main__":
    main()