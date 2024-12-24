import huggingface_hub
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_llm(llm_name, peft=True):
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name)

    # Print model layer names for debugging
    print("Available model layers:")
    for name, _ in model.named_modules():
        print(name)

    # Freeze LLM
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Add PEFT?
    print('Adding LoRA adapters to the model...')
    if True:
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['self_attn.qkv_proj', 'self_attn.o_proj', 'mlp.gate_up_proj', 'mlp.down_proj'])

        model = get_peft_model(model, peft_config)

    return tokenizer, model
