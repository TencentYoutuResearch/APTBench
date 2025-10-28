import os
import argparse
import evaluate
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict
import json
import re


def extract_answer1(response):
    response = response.replace('*', '')
    match = re.search(r'\b([A-Za-z])[\)\n]', response)
    if match:
        return match.group(1)
    else:
        return None


def calculate_rouge(model_output, ground_truth, tokenizer=None):
    """
    计算ROUGE分数
    
    参数:
        model_output (str): 模型生成的文本
        ground_truth (str): 参考文本/真实标签
        tokenizer: 用于分词的tokenizer，如果为None则使用简单的空格分词
        
    返回:
        dict: 包含不同ROUGE分数的字典
    """
    try:
        from rouge import Rouge
    except ImportError:
        print("请先安装rouge包: pip install rouge")
        return {}
    
    # 确保输入是字符串
    if not isinstance(model_output, str) or not isinstance(ground_truth, str):
        model_output = str(model_output)
        ground_truth = str(ground_truth)
    
    # 如果输入为空，返回零分
    if not model_output.strip() or not ground_truth.strip():
        return {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
        }
    
    # 使用tokenizer进行分词（如果提供）
    if tokenizer:
        try:
            # 处理不同类型的tokenizer
            if hasattr(tokenizer, 'tokenize'):
                model_tokens = tokenizer.tokenize(model_output)
                ref_tokens = tokenizer.tokenize(ground_truth)
                model_output = ' '.join(model_tokens)
                ground_truth = ' '.join(ref_tokens)
            elif hasattr(tokenizer, 'encode'):
                model_tokens = tokenizer.encode(model_output)
                ref_tokens = tokenizer.encode(ground_truth)
                if hasattr(tokenizer, 'convert_ids_to_tokens'):
                    model_output = ' '.join(tokenizer.convert_ids_to_tokens(model_tokens))
                    ground_truth = ' '.join(tokenizer.convert_ids_to_tokens(ref_tokens))
        except Exception as e:
            print(f"使用tokenizer时出错: {e}")
            print("使用原始文本计算ROUGE分数")
    
    # 初始化Rouge
    rouge = Rouge()
    
    try:
        # 计算ROUGE分数
        scores = rouge.get_scores(model_output, ground_truth)[0]
        return scores
    except Exception as e:
        print(f"计算ROUGE分数时出错: {e}")
        return {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}
        }


def analyze_folder_per_file(folder_path, tokenizer, mode='ACC', output_filename="total_result.txt"):
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
            if mode == 'EM_with_ROUGE':
                rouge_1 = []
                rouge_2 = []
                rouge_L = []

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 忽略空行
                        if not line.strip():
                            continue
                        
                        total_lines += 1
                        item = json.loads(line)
                        #pred = extract_answer1(item['output'])
                        #if pred == item['answer']:
                        if '"judge": true' in line:
                            judge_true_lines += 1
                        if mode == 'EM_with_ROUGE': 
                            rouge_dict = calculate_rouge(item['pred'], item['answer'], tokenizer)
                            rouge_1.append(rouge_dict['rouge-1']['f']*100)
                            rouge_2.append(rouge_dict['rouge-2']['f']*100)
                            rouge_L.append(rouge_dict['rouge-l']['f']*100)
                if mode == 'EM_with_ROUGE':
                    rouge_1 = sum(rouge_1) / len(rouge_1)
                    rouge_2 = sum(rouge_2) / len(rouge_2)
                    rouge_L = sum(rouge_L) / len(rouge_L)

            except Exception as e:
                print(f"  无法读取文件 {filename}。错误: {e}")
                continue  # 跳过此文件，继续处理下一个

            # 计算当前文件的占比
            if total_lines > 0:
                percentage = (judge_true_lines / total_lines) * 100
            else:
                percentage = 0.0
            if mode == 'ACC':
                # 将格式化的结果添加到列表中
                results.append(f"{filename}: {percentage:.2f}%\n")
            else:
                results.append(f"{filename}: {percentage:.2f}%\t{rouge_1:.2f}%\t{rouge_2:.2f}%\t{rouge_L:.2f}%\n")

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
    tokenizer_path = '../../../assets/llama3_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    if 'tool_call' in args.folder_path:
        mode = 'EM_with_ROUGE'
    else:
        mode = 'ACC'
    
    # 使用传入的文件夹路径调用分析函数
    analyze_folder_per_file(args.folder_path, tokenizer, mode)


if __name__ == "__main__":
    main()
