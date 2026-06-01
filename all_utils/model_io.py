from torch import nn
import torch


def get_model_name(model: nn.Module):
    try:
        model_name = (str(model.model.name_or_path).replace("/", "_"))
    except:
        try:
            model_name = str(model.config._name_or_path).replace("/", "_")
        except:
            try:
                model_name = model.pretrained_cfg.get('architecture', None)
            except:
                model_name = "unknown_model"
    print(f"Model name: {model_name}")
    return model_name


def backup_model(model):
    """
    Creates a CPU backup of a PyTorch model.
    """
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def restore_state_dict(model, backup):
    """
    Restore the state_dict of a model from a CPU backup.
    """
    # Move backup tensors to the same device as the model
    # device = next(model.parameters()).device
    # restored = {k: v.to(device) for k, v in backup.items()}
    model.load_state_dict(backup)
    return model