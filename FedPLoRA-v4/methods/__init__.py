"""FedPLoRA-v4 server aggregation methods.

Each module implements one branch from `branches/`. All methods consume the
v2 fedplora upload package format (state_dict + row_importance + client_size +
client_id + domain), so they can be plugged into `tasks/fed_train_sft_v4.py`
without changing the v2 client-side training loop.
"""
