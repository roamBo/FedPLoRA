from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="code",
        default_dataset="OpenCoder-LLM/opc-sft-stage1",
        description="Prepare code-domain SFT data into FedPLoRA JSONL.",
    )
