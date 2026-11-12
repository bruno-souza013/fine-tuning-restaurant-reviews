import pandas as pd
import torch
from pathlib import Path
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
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

df = pd.read_csv("data/test.csv")
amostra = (
    df.groupby("label", group_keys=False)
    .sample(n=500, random_state=13)
)

def prever(texto):
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
    ).strip().lower()

    return resposta.split()[0].strip(".,!?;:") if resposta else ""


acertos = 0
previsoes_invalidas = 0
rotulos_reais = []
rotulos_previstos = []

for _, exemplo in amostra.iterrows():
    previsao = prever(exemplo["text_pt"])
    rotulos_reais.append(exemplo["label"])
    rotulos_previstos.append(previsao)

    if previsao not in {"negative", "neutral", "positive"}:
        previsoes_invalidas += 1
    elif previsao == exemplo["label"]:
        acertos += 1

total = len(amostra)
print(f"Exemplos avaliados: {total}")
print(f"Acertos: {acertos}")
print(f"Previsões inválidas: {previsoes_invalidas}")
print(f"Acurácia: {acertos / total:.2%}")

rotulos = ["negative", "neutral", "positive"]
rotulos_validos = [
    i for i, previsao in enumerate(rotulos_previstos)
    if previsao in rotulos
]

print("\nRelatório por classe:")
print(
    classification_report(
        [rotulos_reais[i] for i in rotulos_validos],
        [rotulos_previstos[i] for i in rotulos_validos],
        labels=rotulos,
        zero_division=0,
    )
)

print("Matriz de confusão (linhas = real, colunas = previsto):")
print(
    confusion_matrix(
        [rotulos_reais[i] for i in rotulos_validos],
        [rotulos_previstos[i] for i in rotulos_validos],
        labels=rotulos,
    )
)