#verifica se o modelo pode ser carregadoem 4bitsimport torch
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Modelo carregado com sucesso.")
print("Dispositivo:", model.device)