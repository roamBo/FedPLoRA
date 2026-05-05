"""Shared helpers for server-side aggregation (client dict or nn.Module payloads)."""


def client_sd(client_models, i):
    """State dict from a client (nn.Module) or an already-stored dict."""
    m = client_models[i]
    return m.state_dict() if hasattr(m, "state_dict") else m


def obj_sd(obj):
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def upload_package_state(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def upload_package_row_importance(obj):
    if isinstance(obj, dict):
        return obj.get("row_importance", {})
    return {}


def upload_package_client_size(obj):
    if isinstance(obj, dict):
        return obj.get("client_size", None)
    return None


def all_clients_have_key(client_models, key):
    for i in range(len(client_models)):
        if key not in client_sd(client_models, i):
            return False
    return True
