import os
import json
import argparse
import time
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from vllm import LLM, SamplingParams

# --- 全局配置 ---

# SGLang API 服务器地址
URL = "http://127.0.0.1:8001/v1"

# 加载模型路径映射
try:
    model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
except FileNotFoundError:
    print("警告: 'config/model2path.json' 未找到。请确保模型名称是正确的Hugging Face路径或本地路径。")
    model_map = {}


# --- 加载所有Prompt模板 ---
def load_prompt_template(path):
    try:
        return open(path, encoding='utf-8').read()
    except FileNotFoundError:
        print(f"错误: Prompt模板文件 '{path}' 未找到。程序将退出。")
        exit(1)

# swe: env_setup
template_3shot_env_setup_plan = load_prompt_template('prompts/env_setup_plan_3shot.txt')
template_3shot_env_setup_action = load_prompt_template('prompts/env_setup_action_3shot.txt')
template_3shot_env_setup_error = load_prompt_template('prompts/env_setup_error_3shot.txt')

# swe: issue_fix
template_3shot_issue_fix_fix_patch = load_prompt_template('prompts/issue_fix_fix_patch_3shot.txt')
template_3shot_issue_fix_locate = load_prompt_template('prompts/issue_fix_locate_3shot.txt')
template_3shot_issue_fix_plan = load_prompt_template('prompts/issue_fix_plan_3shot.txt')
template_3shot_issue_fix_tool_call = load_prompt_template('prompts/issue_fix_tool_call_3shot.txt')
template_3shot_issue_fix_test_patch = load_prompt_template('prompts/issue_fix_test_patch_3shot.txt')

# deepresearch: close-end
template_3shot_deepresearch_plan_zh = load_prompt_template('prompts/deepresearch_plan_zh_3shot.txt')
template_3shot_deepresearch_plan_en = load_prompt_template('prompts/deepresearch_plan_en_3shot.txt')
template_3shot_deepresearch_summ_ans_zh = load_prompt_template('prompts/deepresearch_summ_ans_zh_3shot.txt')
template_3shot_deepresearch_summ_ans_en = load_prompt_template('prompts/deepresearch_summ_ans_en_3shot.txt')

# deepresearch: open-end
template_3shot_deepresearch_openend_plan_en = load_prompt_template('prompts/deepresearch_openend_plan_en_3shot.txt')
template_3shot_deepresearch_openend_citation_en = load_prompt_template('prompts/deepresearch_openend_citation_en_3shot.txt')
template_3shot_deepresearch_openend_citation_zh = load_prompt_template('prompts/deepresearch_openend_citation_zh_3shot.txt')
template_2shot_deepresearch_openend_quality_en = load_prompt_template('prompts/deepresearch_openend_quality_en_2shot.txt')
template_2shot_deepresearch_openend_quality_zh = load_prompt_template('prompts/deepresearch_openend_quality_zh_2shot.txt')


def truncate_prompt(prompt, tokenizer, max_len=120000):
    """如果prompt过长，则进行截断"""
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        return tokenizer.decode(input_ids, skip_special_tokens=True)
    return prompt


def query_hf(prompt, model_objects, temperature, max_new_tokens):
    """使用Hugging Face Transformers进行推理"""
    model = model_objects['model']
    tokenizer = model_objects['tokenizer']
    
    prompt = truncate_prompt(prompt, tokenizer)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    if temperature == 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["do_sample"] = True

    with torch.no_grad():
        outputs = model.generate(inputs, **generation_kwargs)
    
    generated_tokens = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def query_vllm(prompt, model_objects, temperature, max_new_tokens):
    """使用vLLM进行推理"""
    model = model_objects['model']
    tokenizer = model_objects['tokenizer']

    prompt = truncate_prompt(prompt, tokenizer)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=max_new_tokens
    )
    
    with torch.no_grad():
        outputs = model.generate(prompt, sampling_params)
    
    return outputs[0].outputs[0].text


def query_sglang(prompt, model_objects, temperature, max_new_tokens):
    """通过API使用SGLang进行推理"""
    client = model_objects['client']
    model_name = model_objects['model_name']
    tokenizer = model_objects['tokenizer']

    prompt = truncate_prompt(prompt, tokenizer)

    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].text
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"发生错误: \"{e}\"        正在重试...")
            time.sleep(1)
    else:
        print("已达到最大重试次数，推理失败。")
        return ''


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'\b([A-Za-z])[\)\n]', response)
    return match.group(1) if match else None


def extract_answer_action(response):
    pattern = "\n"
    if pattern in response:
        return response.split(pattern)[0].split(';')[0]
    return response


def extract_answer_summ_ans(response):
    pattern = "]"
    if pattern in response:
        return response.split(pattern)[0]
    return None


def extract_answer_cite(response):
    pattern = ")"
    if pattern in response:
        return response.split(pattern)[0]
    return None


# --- 任务配置字典 ---
# 结构:
# (dataset, task): {
#     "template": 使用的模板,
#     "prompt_fields": { "占位符": "数据项key" },
#     "max_new_tokens": 最大生成token数,
#     "extraction_func": 答案提取函数,
#     "answer_key": 正确答案在数据项中的key,
#     "judge_logic": (预测值, 答案) -> bool, 用于判断对错的逻辑
# }
TASK_CONFIG = {
    ('env_setup', 'plan'): {
        "template": template_3shot_env_setup_plan,
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('env_setup', 'action'): {
        "template": template_3shot_env_setup_action,
        "prompt_fields": {"$EXE_PLAN$": "execution_plan", "$EXE_CMDS$": "executed_cmds"},
        "max_new_tokens": 30,
        "extraction_func": extract_answer_action,
        "answer_key": "target_cmd",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('env_setup', 'error'): {
        "template": template_3shot_env_setup_error,
        "prompt_fields": {"$SETUP$": "setup_instruct", "$ISSUE_TITLE$": "issue_title", "$ISSUE_BODY$": "issue_body", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'plan_zh'): {
        "template": template_3shot_deepresearch_plan_zh,
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'plan_en'): {
        "template": template_3shot_deepresearch_plan_en,
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'summ_ans_zh'): {
        "template": template_3shot_deepresearch_summ_ans_zh,
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory"},
        "max_new_tokens": 30,
        "extraction_func": extract_answer_summ_ans,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'summ_ans_en'): {
        "template": template_3shot_deepresearch_summ_ans_en,
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory"},
        "max_new_tokens": 30,
        "extraction_func": extract_answer_summ_ans,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'openend_plan_en'): {
        "template": template_3shot_deepresearch_openend_plan_en,
        "prompt_fields": {"$QUERY$": "question", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'openend_citation_en'): {
        "template": template_3shot_deepresearch_openend_citation_en,
        "prompt_fields": {"$ARTICLE$": "article", "$CHOICES$": "choices", "$WEB_PAGE$": "url_content"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer_cite,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: set(ans) == (set(pred.split(',')) if pred else set())
    },
    ('deepresearch', 'openend_citation_zh'): {
        "template": template_3shot_deepresearch_openend_citation_zh,
        "prompt_fields": {"$ARTICLE$": "article", "$CHOICES$": "choices", "$WEB_PAGE$": "url_content"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer_cite,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: set(ans) == (set(pred.split(',')) if pred else set())
    },
    ('deepresearch', 'openend_quality_en'): {
        "template": template_2shot_deepresearch_openend_quality_en,
        "prompt_fields": {"$QUERY$": "query", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('deepresearch', 'openend_quality_zh'): {
        "template": template_2shot_deepresearch_openend_quality_zh,
        "prompt_fields": {"$QUERY$": "query", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('issue_fix', 'fix_patch'): {
        "template": template_3shot_issue_fix_fix_patch,
        "prompt_fields": {"$ISSUE$": "issue_statement", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('issue_fix', 'locate'): {
        "template": template_3shot_issue_fix_locate,
        "prompt_fields": {"$ISSUE$": "issue_statement", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('issue_fix', 'plan'): {
        "template": template_3shot_issue_fix_plan,
        "prompt_fields": {"$TRAJ$": "trajs", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('issue_fix', 'tool_call'): {
        "template": template_3shot_issue_fix_tool_call,
        "prompt_fields": {"$TRAJ$": "trajs"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer_action,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    },
    ('issue_fix', 'test_patch'): {
        "template": template_3shot_issue_fix_test_patch,
        "prompt_fields": {"$PROBLEM$": "problem_statement", "$CHOICES$": "choices"},
        "max_new_tokens": 10,
        "extraction_func": extract_answer,
        "answer_key": "answer",
        "judge_logic": lambda pred, ans: pred == ans
    }
}


def get_pred(data, query_func, model_objects, args, fout):
    """
    通用预测函数，根据TASK_CONFIG执行预测。
    
    Args:
        data (list): 待预测的数据列表。
        query_func (function): 用于推理的函数 (e.g., query_hf, query_vllm)。
        model_objects (dict): 包含模型、tokenizer等推理所需对象的字典。
        args (argparse.Namespace): 命令行参数。
        fout (file object): 输出文件对象。
    """
    task_key = (args.dataset, args.task)
    if task_key not in TASK_CONFIG:
        print(f"错误: 任务配置 for {(args.dataset, args.task)} 未在TASK_CONFIG中定义。")
        return

    config = TASK_CONFIG[task_key]
    
    for item in tqdm(data, desc=f"Processing {args.dataset}/{args.task}"):
        # 1. 构建Prompt
        prompt = config['template']
        for placeholder, data_key in config['prompt_fields'].items():
            if data_key in item:
                prompt = prompt.replace(placeholder, item[data_key].strip())
            else:
                print(f"警告: 数据项 {item.get('uuid', 'N/A')} 中缺少key '{data_key}'")

        # 2. 调用指定的推理函数
        output = query_func(
            prompt=prompt,
            model_objects=model_objects,
            temperature=0,
            max_new_tokens=config['max_new_tokens']
        )
        
        if not output:
            continue
        
        output = output.strip()
        
        # 3. 提取预测结果
        pred = config['extraction_func'](output)
        if pred is not None:
            pred = pred.strip()

        # 4. 判断并写入结果
        item['output'] = output
        item['pred'] = pred
        answer = item[config['answer_key']]
        item['judge'] = config['judge_logic'](pred, answer)
        
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results", help="保存结果的目录")
    parser.add_argument("--dataset", "-d", type=str, default="env_setup", help="数据集名称")
    parser.add_argument("--task", "-t", type=str, default="plan", help="任务名称")
    parser.add_argument("--model", "-m", type=str, default="Qwen3-1.7B-Base", help="模型名称或路径")
    parser.add_argument("--backend", "-b", type=str, default="hf", choices=["hf", "vllm", "sglang"], help="选择推理后端: hf, vllm, sglang")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM的张量并行大小")
    args = parser.parse_args()

    print(args)

    # 创建输出目录
    output_dir = os.path.join(args.save_dir, args.dataset, args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    model_name_suffix = args.model.split("/")[-1]
    out_file = os.path.join(output_dir, f"{model_name_suffix}.jsonl")

    # 加载数据
    data_file_path = f'../data/{args.dataset}/{args.task}/input_data.jsonl'
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data_all = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"错误: 数据文件 '{data_file_path}' 未找到。请检查路径。")
        return

    # 缓存处理，避免重复预测
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    has_data[json.loads(line)["uuid"]] = True
                except (json.JSONDecodeError, KeyError):
                    continue # 忽略格式不正确的行
    
    data_to_process = [item for item in data_all if item.get("uuid") not in has_data]
    
    if not data_to_process:
        print("所有数据都已处理过，没有新的数据需要预测。")
        return

    # --- 根据后端初始化模型和分词器 ---
    model_path = model_map.get(args.model, args.model)
    
    print(f"正在从 '{model_path}' 加载Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model_objects = {'tokenizer': tokenizer}
    query_func = None

    print(f"使用后端: {args.backend}")
    if args.backend == 'hf':
        print(f"正在加载模型到Hugging Face Transformers...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        model_objects['model'] = model
        query_func = query_hf
    elif args.backend == 'vllm':
        print(f"正在加载模型到vLLM...")
        model = LLM(
            model=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=131072
        )
        model_objects['model'] = model
        query_func = query_vllm
    elif args.backend == 'sglang':
        print(f"正在初始化SGLang API客户端，目标URL: {URL}")
        client = OpenAI(base_url=URL, api_key="None")
        model_objects['client'] = client
        model_objects['model_name'] = args.model # SGLang API使用模型名
        query_func = query_sglang
    else:
        raise ValueError(f"无效的后端: {args.backend}")

    # --- 执行预测 ---
    with open(out_file, 'a', encoding='utf-8') as fout:
        get_pred(data_to_process, query_func, model_objects, args, fout)
    
    print(f"预测完成，结果已保存到: {out_file}")

if __name__ == "__main__":
    main()