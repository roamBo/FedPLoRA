from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="finance",
        default_dataset="gbharti/finance-alpaca",
        description="Prepare finance-domain SFT data into FedPLoRA JSONL.",
    )
