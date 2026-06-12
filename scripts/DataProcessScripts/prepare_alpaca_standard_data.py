"""
Export Stanford Alpaca (tatsu-lab/alpaca) into FedPLoRA unified JSONL.

Prompt template follows the classic Alpaca format:
  instruction + optional input -> response (output field).
"""
from prepare_hf_domain_data import main


if __name__ == "__main__":
    main(
        default_domain="alpaca",
        default_dataset="tatsu-lab/alpaca",
        description="Prepare Stanford Alpaca into standard (non-cross-domain) SFT JSONL.",
    )
