import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

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
        "content": (
            "A comida estava deliciosa, o atendimento foi rápido "
            "e o ambiente era agradável."
        ),
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
)
print("Resposta do modelo:", repr(resposta.strip()))