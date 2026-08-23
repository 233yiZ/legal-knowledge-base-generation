#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legal Knowledge Generation Script
调用大模型API为每个罪名生成结构化法律知识
模型: Qwen3-235B-A22B (通过DashScope或OpenAI兼容接口)
"""

import os
import json
import time
import argparse
from typing import List, Dict, Optional

# ============ 配置 ============
# 使用阿里云DashScope SDK (推荐)
# pip install dashscope
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("Warning: dashscope not installed. Install with: pip install dashscope")

# ============ 参数解析 ============
def parse_args():
    parser = argparse.ArgumentParser(description="Generate legal knowledge via LLM API")
    parser.add_argument("--charges_file", type=str, default="charges.txt",
                        help="Path to file containing charge names, one per line")
    parser.add_argument("--template_file", type=str, default="prompt_template.txt",
                        help="Path to prompt template file with [xxx] placeholder")
    parser.add_argument("--output", type=str, default="knowledge_base.json",
                        help="Output JSON file path")
    parser.add_argument("--api_type", type=str, default="dashscope", choices=["dashscope", "openai"],
                        help="API type: dashscope or openai")
    parser.add_argument("--model", type=str, default="qwen3-235b-a22b",
                        help="Model name (for dashscope: qwen-max, qwen-plus, etc.; for openai: model id)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (if not set, read from environment variable)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for OpenAI-compatible API (optional)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay in seconds between API calls to avoid rate limiting")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries for failed requests")
    return parser.parse_args()


# ============ 提示模板处理 ============
def load_prompt_template(file_path: str) -> str:
    """读取提示模板，应包含 [xxx] 占位符"""
    with open(file_path, 'r', encoding='utf-8') as f:
        template = f.read().strip()
    if "[xxx]" not in template:
        raise ValueError("Prompt template must contain '[xxx]' placeholder for charge name.")
    return template


def build_prompt(template: str, charge: str) -> str:
    """替换占位符生成完整提示"""
    return template.replace("[xxx]", charge)


# ============ 调用 API ============
def call_dashscope(charge: str, prompt: str, model: str, api_key: str) -> Optional[str]:
    """使用DashScope调用Qwen模型"""
    if not DASHSCOPE_AVAILABLE:
        raise ImportError("dashscope not installed. Please install: pip install dashscope")
    
    dashscope.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        raise ValueError("API key not provided. Set DASHSCOPE_API_KEY environment variable or pass --api_key.")

    try:
        response = Generation.call(
            model=model,  # e.g., "qwen-max", "qwen-plus", "qwen-turbo"
            prompt=prompt,
            temperature=0.1,
            max_tokens=1024,
            top_p=0.8,
            stop=None,
            result_format="message"
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            print(f"Error for charge '{charge}': {response.status_code} - {response.message}")
            return None
    except Exception as e:
        print(f"Exception for charge '{charge}': {e}")
        return None



def generate_knowledge_for_charge(charge: str, template: str, args) -> Dict:
    """为单个罪名生成知识，返回包含状态和内容的字典"""
    prompt = build_prompt(template, charge)
    print(f"Generating for: {charge}")

    for attempt in range(1, args.max_retries + 1):
        if args.api_type == "dashscope":
            result = call_dashscope(charge, prompt, args.model, args.api_key)
        else:
            print("Error for LLM api connection")
        
        if result is not None:
            return {
                "charge": charge,
                "success": True,
                "content": result.strip()
            }
        else:
            print(f"Attempt {attempt}/{args.max_retries} failed for '{charge}'. Retrying...")
            time.sleep(2 ** attempt)  # 指数退避
    
    return {
        "charge": charge,
        "success": False,
        "content": None,
        "error": "Max retries exceeded"
    }


# ============ 主流程 ============
def main():
    args = parse_args()

    # 读取罪名列表
    with open(args.charges_file, 'r', encoding='utf-8') as f:
        charges = [line.strip() for line in f if line.strip()]

    if not charges:
        print("No charges found in input file.")
        return

    print(f"Loaded {len(charges)} charges.")

    # 读取提示模板
    template = load_prompt_template(args.template_file)

    # 存储结果
    results = []
    failed = []

    # 逐个生成
    for idx, charge in enumerate(charges, 1):
        print(f"\n[{idx}/{len(charges)}] Processing: {charge}")
        result = generate_knowledge_for_charge(charge, template, args)
        results.append(result)
        if not result["success"]:
            failed.append(charge)

        # 保存中间结果（防止中断丢失数据）
        with open(args.output + ".tmp", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 延迟
        if idx < len(charges):
            time.sleep(args.delay)

    # 最终输出
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Output saved to {args.output}")
    if failed:
        print(f"Failed charges ({len(failed)}): {', '.join(failed)}")

    # 打印统计
    success_count = sum(1 for r in results if r["success"])
    print(f"Success: {success_count}/{len(charges)}")


if __name__ == "__main__":
    main()