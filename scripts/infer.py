from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = Path("outputs/qlora-adapter")


def load_model_and_tokenizer():
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"O adaptador não foi encontrado em '{ADAPTER_PATH}'. "
            "Execute primeiro o treinamento em scripts/train_qlora.py."
        )

    tokenizer_source = ADAPTER_PATH if (ADAPTER_PATH / "tokenizer_config.json").exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str):
    formatted_prompt = f"### Instrução:\n{prompt}\n\n### Resposta:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def main():
    model, tokenizer = load_model_and_tokenizer()

    prompt = "Quero agendar um corte para amanhã à tarde."
    response = generate_response(model, tokenizer, prompt)

    print("Prompt:")
    print(prompt)
    print("\nResposta gerada:")
    print(response)


if __name__ == "__main__":
    main()