import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def formatar_exemplo(exemplo):
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
            "content": exemplo["text_pt"],
        },
        {
            "role": "assistant",
            "content": exemplo["label"],
        },
    ]

    return {
        "text": tokenizer.apply_chat_template(
            mensagens,
            tokenize=False,
            add_generation_prompt=False,
        )
    }

df = pd.read_csv("data/train.csv")

dataset = Dataset.from_pandas(df[["text_pt", "label"]])
dataset_formatado = dataset.map(formatar_exemplo)

print(dataset_formatado)
print("\nPrimeiro exemplo:")
print(dataset_formatado[0]["text"])