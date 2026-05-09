import argparse
import json
import os


DEFAULT_DOMAINS = [
    "general",
    "math",
    "code",
    "medical",
    "legal",
    "finance",
    "education",
]


DEFAULT_SAMPLES = {
    "general": [
        {
            "prompt": "请概括远程协作中提高沟通效率的三个原则。",
            "response": "第一，明确任务边界和产出标准；第二，采用统一的文档与沟通接口；第三，及时同步风险与阻塞项。",
        }
    ],
    "math": [
        {
            "prompt": "求解方程 x^2-5x+6=0，并说明步骤。",
            "response": "先因式分解得到 (x-2)(x-3)=0，因此解为 x=2 或 x=3。",
        }
    ],
    "code": [
        {
            "prompt": "写一个 Python 函数，返回列表中的最大值。",
            "response": "```python\ndef find_max(nums):\n    if not nums:\n        raise ValueError('nums must not be empty')\n    return max(nums)\n```",
        }
    ],
    "medical": [
        {
            "prompt": "解释高血压患者常见的生活方式干预建议。",
            "response": "常见建议包括限盐饮食、控制体重、规律运动、减少饮酒与戒烟；具体方案应遵医嘱执行。",
        }
    ],
    "legal": [
        {
            "prompt": "合同审查时通常需要重点关注哪些条款？",
            "response": "通常需要关注主体资格、权利义务、违约责任、争议解决、保密与知识产权等核心条款。",
        }
    ],
    "finance": [
        {
            "prompt": "解释资产配置中分散化的基本作用。",
            "response": "分散化的核心作用是降低单一资产波动对组合收益的影响，在给定风险偏好下改善风险收益比。",
        }
    ],
    "education": [
        {
            "prompt": "如何针对基础较弱的学生设计一次有效的反馈？",
            "response": "反馈应具体、可执行、分步骤，并先肯定已完成部分，再指出下一步最关键的改进点。",
        }
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Create a 7-domain JSONL template for FedPLoRA domain benchmark."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/domain_7_template.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(DEFAULT_DOMAINS),
        help="Comma-separated domain names.",
    )
    parser.add_argument(
        "--examples_per_domain",
        type=int,
        default=1,
        help="Number of template examples per domain.",
    )
    args = parser.parse_args()

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    rows = []
    for domain in domains:
        seed_examples = DEFAULT_SAMPLES.get(domain, [])
        for idx in range(max(args.examples_per_domain, 1)):
            if idx < len(seed_examples):
                example = dict(seed_examples[idx])
            else:
                example = {
                    "prompt": f"请补充一个 {domain} 领域的指令样本。",
                    "response": f"这里填写 {domain} 领域的标准回答。",
                }
            rows.append(
                {
                    "domain": domain,
                    "prompt": example["prompt"],
                    "response": example["response"],
                    "source_id": f"{domain}_template_{idx:04d}",
                    "metadata": {
                        "dataset": "template",
                        "split": "raw",
                    },
                }
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] wrote template jsonl to {args.output}")
    print(f"[ok] domains={domains} rows={len(rows)}")


if __name__ == "__main__":
    main()
