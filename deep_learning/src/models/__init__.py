from .convnext import ConvNext
from .efficientnet_pytorch import EfficientNetPytorch
from .efficientnet import EfficientNet
from .google_vision_transformer import GoogleVisionTransformer
from .vision_transformer import VisionTransformer
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2


def get_model(model_name, num_model_number, num_category, num_color):

    if "tf_efficientnet" in model_name:
        model = EfficientNet(model_name, num_model_number, num_category, num_color)

    elif "efficientnet" in model_name:
        model = EfficientNetPytorch(
            model_name, num_model_number, num_category, num_color
        )

    elif "google/vit" in model_name:
        model = GoogleVisionTransformer(
            model_name, num_model_number, num_category, num_color
        )

    elif "vit" in model_name:
        model = VisionTransformer(model_name, num_model_number, num_category, num_color)

    elif "swinv2" in model_name:
        model = SwinTransformerV2(model_name, num_model_number, num_category, num_color)

    elif "swin" in model_name:
        model = SwinTransformer(model_name, num_model_number, num_category, num_color)

    elif "convnext" in model_name:
        model = ConvNext(model_name, num_model_number, num_category, num_color)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
