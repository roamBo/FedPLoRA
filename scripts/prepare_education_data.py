from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="education",
        default_dataset="ScaleAI/TutorBench",
        description="Prepare education-domain SFT data into FedPLoRA JSONL.",
    )
