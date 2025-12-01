# run_gsm8k_ttt.py
import argparse
import os
from datasets import load_dataset

import sglang
from sglang.srt.utils import kill_process_tree


def main(args):
    print("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main", split="train")
    if args.num_samples > 0:
        ds = ds.select(range(args.num_samples))
    print(f"Total samples: {len(ds)}")

    print("Initializing SGLang Engine with Online TTT...")
    engine = sglang.Engine(
        model_path=args.model_path,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=args.draft_model_path,
        speculative_num_draft_tokens=64,
        speculative_eagle_topk=8,
        speculative_num_steps=4,
        enable_online_ttt=True,
        online_ttt_interval=args.interval,
        online_ttt_lr=1e-5,
        tp_size=1,
        mem_fraction_static=0.7,
        disable_cuda_graph=False,
        online_ttt_device="cuda:1"
    )

    # 简单的 prompt 模板（zero-shot CoT）
    template = "Question: {question}\nLet's think step by step.\nAnswer:"

    print("Starting Inference Loop...")

    try:
        for i, item in enumerate(ds):
            question = item["question"]
            prompt = template.format(question=question)

            print(f"\n[Request {i+1}/{len(ds)}] Processing...")
            # 根据 sglang 版本，generate API 可能略有不同
            out = engine.generate(
                prompt=prompt,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 512,
                },
            )

            # out 可能是 dict 或 list，这里做个兼容
            text = None
            if isinstance(out, dict):
                text = out.get("text", None)
            elif isinstance(out, list) and len(out) > 0:
                # 有些版本返回一个结果列表
                first = out[0]
                text = getattr(first, "text", None) or first.get("text", None)

            if text is None:
                print("Output: [unknown format]")
            else:
                print(f"Output: {text[:80]}...")

    finally:
        print("Shutting down...")
        engine.shutdown()
        kill_process_tree(os.getpid())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to run (-1 for all)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="TTT update interval (in requests)",
    )


    args = parser.parse_args()

    # WandB 环境变量（也可以在外面 export）
    os.environ.setdefault("WANDB_PROJECT", "eagle3-ttt-test")
    # os.environ["WANDB_API_KEY"] = "你的WANDB_KEY"

    main(args)
