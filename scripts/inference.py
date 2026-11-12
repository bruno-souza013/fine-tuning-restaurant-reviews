import torch
from peft import PeftModel
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
ADAPTER = ROOT / "qlora-restaurant-reviews"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

print("Modelo carregado.")

def classificar(texto):
    mensagens = [
        {
            "role": "system",
            "content": (
                "Classifique avaliações de restaurantes usando apenas: "
                "negative, neutral ou positive. Responda somente com um rótulo."
            ),
        },
        {"role": "user", "content": texto},
    ]

    entrada = tokenizer.apply_chat_template(
        mensagens,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        saida = model.generate(
            **entrada,
            max_new_tokens=5,
            do_sample=False,
        )

    resposta = tokenizer.decode(
        saida[0][entrada["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return resposta


print("\nDigite uma avaliação de restaurante.")
print("Digite 'sair' para encerrar.\n")

while True:
    texto = input("Avaliação: ").strip()

    if texto.lower() == "sair":
        print("Encerrando.")
        break

    if not texto:
        print("Digite uma avaliação válida.")
        continue

    classificacao = classificar(texto)
    print(f"Classificação: {classificacao}\n")