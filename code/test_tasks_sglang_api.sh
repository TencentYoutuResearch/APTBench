#!/bin/bash
# -*- coding: utf-8 -*-
model=$1
# 定义datasets列表
datasets=("env_setup" "deepresearch" "issue_fix")

# 外层循环：遍历datasets
for dataset in "${datasets[@]}"; do    
    # 根据当前dataset确定tasks
    case "$dataset" in
        "env_setup")
            tasks=("plan" "action" "error")
            ;;
        "deepresearch")
            tasks=("plan_en" "plan_zh" "summ_ans_en" "summ_ans_zh" "openend_plan_en" "openend_citation_en" "openend_citation_zh" "openend_quality_en" "openend_quality_zh")
            ;;
        "issue_fix")
            tasks=("locate" "fix_patch" "plan" "action" "test_patch")
            ;;
        *)
            echo "未知的dataset: $dataset"
            continue
            ;;
    esac
    
    for task in "${tasks[@]}"; do
        echo "正在推理 dataset: $dataset, task: $task"
        python pred_sglang_api.py -t $task -d $dataset -m $model -b "sglang"
    done
done

for dataset in "${datasets[@]}"; do    
    # 根据当前dataset确定tasks
    case "$dataset" in
        "env_setup")
            tasks=("plan" "action" "error")
            ;;
        "deepresearch")
            tasks=("plan_en" "plan_zh" "summ_ans_en" "summ_ans_zh" "openend_plan_en" "openend_citation_en" "openend_citation_zh" "openend_quality_en" "openend_quality_zh")
            ;;
        "issue_fix")
            tasks=("locate" "fix_patch" "plan" "action" "test_patch")
            ;;
        *)
            echo "未知的dataset: $dataset"
            continue
            ;;
    esac
    cd results/$dataset
    for task in "${tasks[@]}"; do
        python get_results.py -f $task
        echo -e "----------${dataset}-${task} results:---------"
        cat $task/total_result.txt
    done
    cd ../../
done

echo "所有任务处理完毕"
