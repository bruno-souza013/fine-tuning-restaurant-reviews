import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

df_teste = pd.read_csv("data/test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.eval()

rotulos_reais = []
rotulos_preditos = []
rotulos_validos = {"negative", "neutral", "positive"}

for _, exemplo in df_teste.iterrows():
    texto = exemplo["text_pt"]
    rotulo_real = exemplo["label"]

    mensagens = [
        {
            "role": "system",
            "content": (
                "Classifique avaliações de restaurantes usando apenas: "
                "negative, neutral ou positive."
            ),
        },
        {
            "role": "user",
            "content": texto,
        },
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
            max_new_tokens=10,
            do_sample=False,
        )

    resposta = tokenizer.decode(
        saida[0][entrada["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    resposta = resposta.strip().lower()

    if resposta not in rotulos_validos:
        resposta = "invalid"

    rotulos_reais.append(rotulo_real)
    rotulos_preditos.append(resposta)

    print("Real:", rotulo_real)
    print("Predito:", repr(resposta))
    print("-" * 40)

print("\nAccuracy:", accuracy_score(rotulos_reais, rotulos_preditos))
classes = sorted(rotulos_validos) + ["invalid"]
respostas_invalidas = rotulos_preditos.count("invalid")
print("Respostas inválidas:", respostas_invalidas)

print("\nRelatório de classificação:")
print(
    classification_report(
        rotulos_reais,
        rotulos_preditos,
        labels=sorted(rotulos_validos),
        zero_division=0,
    )
)

print("Matriz de confusão:")
matriz = confusion_matrix(
    rotulos_reais,
    rotulos_preditos,
    labels=classes,
)
print(matriz)
print("Ordem das classes:", classes)

Path("outputs").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(matriz, cmap="Blues")
ax.set(
    xticks=range(len(classes)),
    yticks=range(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Rótulo predito",
    ylabel="Rótulo real",
    title="Matriz de confusão - baseline",
)

for linha in range(len(classes)):
    for coluna in range(len(classes)):
        ax.text(coluna, linha, matriz[linha, coluna], ha="center", va="center")

fig.tight_layout()
fig.savefig("outputs/baseline_confusion_matrix.png", dpi=150)
plt.close(fig)
print("Matriz salva em: outputs/baseline_confusion_matrix.png")