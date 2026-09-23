import random
import torch
from PIL import Image
from models.base import BaseVLMModel


class SmolVLM(BaseVLMModel):
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-Instruct", device: str | None = None, mock: bool = False):
        super().__init__(model_name=model_name, device=device, mock=mock)

        if not self.mock:
            from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.processor = AutoProcessor.from_pretrained(model_name)
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                )
            except Exception:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                )
            self.model.to(self.device)
            self.model.eval()

    def _prepare_inputs(self, image: Image.Image | str, prompt: str):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding=True).to(self.device)
        return inputs

    def generate(self, image: Image.Image | str, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0):
        if self.mock:
            # Deterministic mock response based on prompt text or image size
            seed = len(prompt) + (image.size[0] if isinstance(image, Image.Image) else len(str(image)))
            rng = random.Random(seed)
            conf_val = rng.randint(55, 95)
            labels = ["pizza", "hamburger", "sushi", "tacos", "caesar_salad", "ice_cream", "ramen"]
            predicted = rng.choice(labels)
            answer_text = f"I am {conf_val}% confident this is {predicted}."
            token_probs = [min(1.0, max(0.1, (conf_val / 100.0) + rng.uniform(-0.05, 0.05))) for _ in range(5)]
            return answer_text, token_probs

        inputs = self._prepare_inputs(image, prompt)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs.sequences[0, input_length:]
        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        token_probs = []
        if outputs.scores is not None:
            for score_step, token_id in zip(outputs.scores, generated_ids):
                probs = torch.softmax(score_step[0], dim=-1)
                token_probs.append(probs[int(token_id)].item())

        return answer_text, token_probs
