#!/usr/bin/env python3
from omniplus.inference import PaliGemmaInference
from omniplus.utils.logger import setup_logger
from omniplus.utils.timer import cuda_timer


def main(
    model_path: str = "paligemma-3b-pt-448",
    image_path: str = "test.jpg",
    prompt: str = "What does it say on the t-shirt?",
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    logger = setup_logger("TestScript")
    model = PaliGemmaInference(only_cpu=only_cpu)

    # Load model silently
    model.load_model(model_path)

    # Generate with timing
    timer = cuda_timer()
    prompt, generated = model.generate(
        prompt=prompt,
        image_file_path=image_path,
        max_tokens_to_generate=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample)
    breakpoint()
    # Log generation result and time
    logger.info(f"Generated: {generated}")
    logger.info(f"Time: {timer()}")


if __name__ == "__main__":
    main()
