from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="medical",
        default_dataset="FreedomIntelligence/medical-o1-reasoning-SFT",
        description="Prepare medical-domain SFT data into FedPLoRA JSONL.",
    )
