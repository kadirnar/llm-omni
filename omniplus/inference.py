from typing import Tuple

import torch
from PIL import Image

from omniplus.model.modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from omniplus.model.modeling_paligemma import load_hf_model
from omniplus.model.processing_paligemma import PaliGemmaProcessor


class PaliGemmaInference:

    def __init__(self, only_cpu: bool = False):
        self.device = self._get_device(only_cpu)
        self.model = None
        self.processor = None

    def _get_device(self, only_cpu: bool) -> str:
        if not only_cpu:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def load_model(self, model_path: str) -> None:
        model, tokenizer = load_hf_model(model_path, self.device)
        self.model = model.to(self.device).eval()

        num_image_tokens = model.config.vision_config.num_image_tokens
        image_size = model.config.vision_config.image_size
        self.processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    def _move_inputs_to_device(self, model_inputs: dict) -> dict:
        return {k: v.to(self.device) for k, v in model_inputs.items()}

    def _get_model_inputs(self, prompt: str, image_file_path: str) -> dict:
        image = Image.open(image_file_path)
        images = [image]
        prompts = [prompt]
        model_inputs = self.processor(text=prompts, images=images)
        model_inputs = self._move_inputs_to_device(model_inputs)
        return model_inputs

    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def generate(
        self,
        prompt: str,
        image_file_path: str,
        max_tokens_to_generate: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> Tuple[str, str]:
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model first.")

        model_inputs = self._get_model_inputs(prompt, image_file_path)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        kv_cache = KVCache()
        stop_token = self.processor.tokenizer.eos_token_id
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_tokens_to_generate):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
                kv_cache = outputs["kv_cache"]
                next_token_logits = outputs["logits"][:, -1, :]

                if do_sample:
                    next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = self._sample_top_p(next_token_logits, top_p)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                next_token = next_token.squeeze(0)
                generated_tokens.append(next_token)

                if next_token.item() == stop_token:
                    break

                input_ids = next_token.unsqueeze(-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)

            generated_tokens = torch.cat(generated_tokens, dim=-1)
            decoded = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return prompt, decoded
