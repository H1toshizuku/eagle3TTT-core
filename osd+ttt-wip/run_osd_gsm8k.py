import argparse
import asyncio
import logging
import os
import sys
from typing import List

# 尝试导入 datasets
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install datasets: pip install datasets")

import sglang
from sglang.utils import print_highlight

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ttt_runner")

def get_gsm8k_prompts(split: str = "train", num_samples: int = -1) -> List[str]:
    """
    加载 GSM8K 数据集并构建适合 Qwen 的 ChatML 格式 Prompt。
    """
    print_highlight(f"Loading GSM8K [{split}] dataset...")
    try:
        dataset = load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    prompts = []
    # Qwen ChatML Template
    # 系统提示词有助于稳定 CoT 输出
    system_prompt = "You are a helpful assistant. Please solve the math problem step by step."
    template = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    for i, item in enumerate(dataset):
        if num_samples > 0 and i >= num_samples: 
            break
        prompt = template.format(system=system_prompt, question=item['question'])
        prompts.append(prompt)
    
    print_highlight(f"Loaded {len(prompts)} prompts.")
    return prompts

async def run_requests(engine, prompts, parallelism=1):
    """
    异步发送请求
    """
    semaphore = asyncio.Semaphore(parallelism)
    
    async def send_one(prompt, index):
        async with semaphore:
            # EAGLE-3 推荐配置
            # Temperature > 0 有助于覆盖更多 Draft 树路径，提供更丰富的 TTT 训练样本
            sampling_params = {
                "temperature": 0.7, 
                "max_new_tokens": 512,
                "top_p": 0.9
            }
            try:
                # 使用 async_generate 非阻塞调用
                await engine.async_generate(prompt=prompt, sampling_params=sampling_params)
                
                if (index + 1) % 10 == 0: 
                    logger.info(f"Finished request {index + 1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"Request {index} failed: {e}")

    tasks = [send_one(p, i) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description="Run EAGLE-3 Online TTT with Qwen3")
    
    # === 模型路径 ===
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model-path", type=str, default="Tengyunw/qwen3_8b_eagle3")
    
    # === EAGLE-3 核心参数 (严格遵循官方配置) ===
    parser.add_argument("--speculative-algorithm", type=str, default="EAGLE3", help="Must be EAGLE3")
    parser.add_argument("--speculative-num-steps", type=int, default=6, help="Draft depth")
    parser.add_argument("--speculative-eagle-topk", type=int, default=10, help="Draft tree width")
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=64, help="Total draft tokens")
    
    # === Online TTT 超参数 (新功能) ===
    # parser.add_argument("--enable-online-ttt", action="store_true", help="Enable TTT")
    parser.add_argument("--ttt-lr", type=float, default=1e-5, help="Learning rate for shadow trainer")
    parser.add_argument("--ttt-interval", type=int, default=1024, help="Update weights every N requests")
    parser.add_argument("--ttt-device", type=str, default="cuda:1", help="Device for Shadow Trainer")
    parser.add_argument("--wandb-project", type=str, default="qwen3-eagle3-ttt", help="WandB project name")
    
    # === 系统配置 ===
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--parallelism", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.6, help="Reserved memory fraction for KV cache")

    args = parser.parse_args()

    # 设置环境变量供子进程 (Scheduler/Worker) 使用
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print_highlight("Initializing SGLang Engine for EAGLE-3 + Online TTT...")
    
    engine = sglang.Engine(
        # 基础配置
        model_path=args.model_path,
        tp_size=args.tp_size,
        trust_remote_code=True,
        mem_fraction_static=args.mem_fraction_static,
        log_level="info",
        
        # 显存/图优化 (EAGLE 推荐配置)
        cuda_graph_max_bs=8, 
        dtype="bfloat16",
        
        # Speculative Decoding 配置
        speculative_algorithm=args.speculative_algorithm,
        speculative_draft_model_path=args.draft_model_path,
        speculative_eagle_topk=args.speculative_eagle_topk, 
        speculative_num_steps=args.speculative_num_steps,
        speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        
        # Online TTT 配置 (对应 server_args.py 的修改)
        enable_online_ttt=True,
        online_ttt_interval=args.ttt_interval,
        online_ttt_lr=args.ttt_lr,
        online_ttt_device=args.ttt_device,
    )

    prompts = get_gsm8k_prompts(num_samples=args.num_samples)

    try:
        print_highlight(f"Starting inference with {args.parallelism} concurrent requests...")
        asyncio.run(run_requests(engine, prompts, args.parallelism))
    except KeyboardInterrupt:
        print_highlight("Interrupted by user.")
    finally:
        print_highlight("Shutting down engine...")
        engine.shutdown()

if __name__ == "__main__":
    main()