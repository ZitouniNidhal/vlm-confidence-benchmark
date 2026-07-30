import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer


class SmolVLM:
    def __init__(self, model_name: str = "HuggingFace/smolvlm-instruct", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(model_name)
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
        inputs = self._prepare_inputs(image, prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                output_scores=True,
                return_dict_in_generate=True,
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
