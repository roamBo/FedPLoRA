from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="general",
        default_dataset="allenai/tulu-3-sft-mixture",
        description="Prepare general-domain SFT data into FedPLoRA JSONL.",
    )
