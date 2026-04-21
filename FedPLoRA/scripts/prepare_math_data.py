from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="math",
        default_dataset="AI-MO/NuminaMath-CoT",
        description="Prepare math-domain SFT data into FedPLoRA JSONL.",
    )
