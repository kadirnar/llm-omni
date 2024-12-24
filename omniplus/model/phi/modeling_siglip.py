import requests
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def get_image_encoder(image_encoder_str, peft=False):
    processor = ViTImageProcessor.from_pretrained(image_encoder_str)
    model = ViTForImageClassification.from_pretrained(image_encoder_str)

    if peft:

        from peft import LoraConfig, TaskType, get_peft_config, get_peft_model

        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['dense'])

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        model.requires_grad = False

    return processor, model
