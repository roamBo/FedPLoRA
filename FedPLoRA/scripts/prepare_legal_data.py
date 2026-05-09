from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="legal",
        default_dataset="lawinstruct/lawinstruct",
        description="Prepare legal-domain SFT data into FedPLoRA JSONL.",
    )
