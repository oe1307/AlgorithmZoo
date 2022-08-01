def mean_std(model_name):

    # normalize時のmean, std
    if "efficientnet" in model_name:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif "tf_efficientnet" in model_name:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif "tf_efficientnetv2" in model_name:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif "google/vit" in model_name:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif "vit" in model_name:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif "swinv2" in model_name:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif "swin" in model_name:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif "convnext" in model_name:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return mean, std
