def rebuild_from_payload(payload):
    """Rebuild a model from a saved payload alone. Used to prove the save is sufficient."""
    from pilot.model_masked_v1 import MaskedSignTCN

    model = MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                          kernel_size=payload["kernel_size"], dropout=payload["dropout"],
                          n_classes=payload["n_classes"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, list(payload["classes"])