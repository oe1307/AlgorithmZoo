model2size = (
    {
        # _EfficientNet
        "efficientnet-b0": 224,
        "efficientnet-b1": 240,
        "efficientnet-b2": 260,
        "efficientnet-b3": 300,
        "efficientnet-b4": 380,
        "efficientnet-b5": 456,
        "efficientnet-b6": 528,
        "efficientnet-b7": 600,
    }
    | {
        # EffientNetV2
        "tf_efficientnet_b0_ns": 224,
        "tf_efficientnet_b1_ns": 240,
        "tf_efficientnet_b2_ns": 260,
        "tf_efficientnet_b3_ns": 300,
        "tf_efficientnet_b4_ns": 380,
        "tf_efficientnet_b5_ns": 456,
        "tf_efficientnet_b6_ns": 528,
        "tf_efficientnet_b7_ns": 600,
    }
    | {
        # EfficientNet
        "tf_efficientnetv2_b0": 192,
        "tf_efficientnetv2_b1": 192,
        "tf_efficientnetv2_b2": 208,
        "tf_efficientnetv2_b3": 240,
        "tf_efficientnetv2_s": 300,
        "tf_efficientnetv2_m": 384,
        "tf_efficientnetv2_l": 384,
        "tf_efficientnetv2_xl": 384,
    }
    | {
        # GoogleVisionTransformer
        "google/vit-base-patch16-224": 224,
        "google/vit-base-patch32-384": 384,
        "google/vit-large-patch16-224": 224,
        "google/vit-large-patch32-384": 384,
        "google/vit-huge-patch14-224": 224,
    }
    | {
        # VisionTransformer
        "vit_tiny_patch16_224": 224,
        "vit_tiny_patch16_384": 384,
        "vit_small_patch32_224": 224,
        "vit_small_patch32_384": 384,
        "vit_small_patch16_224": 224,
        "vit_small_patch16_384": 384,
        "vit_base_patch32_224": 224,
        "vit_base_patch32_384": 384,
        "vit_base_patch16_224": 224,
        "vit_base_patch16_384": 384,
    }
    | {
        # SwinTransformer
        "swin_tiny_patch4_window7_224": 224,
        "swin_small_patch4_window7_224": 224,
        "swin_base_patch4_window12_384": 384,
        "swin_base_patch4_window7_224": 224,
        "swin_large_patch4_window12_384": 384,
        "swin_large_patch4_window7_224": 224,
        "swin_large_patch4_window7_224": 224,
    }
    | {
        # SwinTransformerV2
        "swinv2_small_window8_256": 256,
        "swinv2_small_window16_256": 256,
        "swinv2_base_window8_256": 256,
        "swinv2_base_window16_256": 256,
        "swinv2_large_window12_256": 256,
        "swinv2_large_window16_256": 256,
    }
)
