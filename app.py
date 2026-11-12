import os
from functools import lru_cache

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "brunoSouza013/restaurant-review-qlora-adapter"
HF_TOKEN = os.environ["HF_TOKEN"]
VALID_LABELS = {"negative", "neutral", "positive"}


def validar_avaliacao(texto: str) -> str | None:
    texto = texto.strip()
    if not texto:
        return "Digite uma avaliação."

    if len(texto) < 3:
        return "Digite uma avaliação com pelo menos 3 caracteres."

    return None


@lru_cache(maxsize=1)
def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_REPO,
        token=HF_TOKEN,
    )

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
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_REPO,
        token=HF_TOKEN,
    )
    model.eval()
    return tokenizer, model


def gerar_resposta(tokenizer, model, mensagens, max_new_tokens):
    entrada = tokenizer.apply_chat_template(
        mensagens,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        saida = model.generate(
            **entrada,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(
        saida[0][entrada["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip().casefold()


def texto_eh_avaliacao(tokenizer, model, texto):
    mensagens = [
        {
            "role": "system",
            "content": (
                "Você é um filtro de domínio. Responda somente com "
                "RELEVANT ou IRRELEVANT."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classifique como RELEVANT se o texto for uma avaliação, "
                "opinião ou experiência sobre restaurante, comida, bebida, "
                "atendimento ou ambiente. Caso contrário, responda "
                "IRRELEVANT.\n\nTexto:\n"
                f"{texto}"
            ),
        },
    ]
    resposta = gerar_resposta(tokenizer, model, mensagens, max_new_tokens=3)
    return resposta.startswith("relevant")


@spaces.GPU
def classificar_avaliacao(texto: str) -> str:
    erro = validar_avaliacao(texto)
    if erro:
        return erro

    tokenizer, model = carregar_modelo()
    if not texto_eh_avaliacao(tokenizer, model, texto.strip()):
        return (
            "Este texto não parece ser uma avaliação de restaurante. "
            "Descreva sua experiência com a comida, o atendimento ou o ambiente."
        )

    mensagens = [
        {
            "role": "system",
            "content": (
                "Classifique avaliações de restaurantes usando apenas: "
                "negative, neutral ou positive. Responda somente com um rótulo."
            ),
        },
        {"role": "user", "content": texto.strip()},
    ]
    resposta = gerar_resposta(tokenizer, model, mensagens, max_new_tokens=5)
    for rotulo in VALID_LABELS:
        if rotulo in resposta:
            return rotulo
    return "Não foi possível classificar essa avaliação."


CSS = """
.gradio-container { max-width: 950px !important; margin: auto; }
#titulo, #subtitulo { text-align: center; }
#subtitulo { color: #64748b; font-size: 1.1rem; }
#resultado textarea { font-size: 1.4rem !important; text-align: center; font-weight: bold; }
"""

with gr.Blocks(css=CSS, title="Restaurant Review QLoRA") as interface:
    gr.Markdown("# Classificador de avaliações", elem_id="titulo")
    gr.Markdown(
        "Descubra se uma avaliação é positiva, neutra ou negativa.",
        elem_id="subtitulo",
    )

    with gr.Row():
        with gr.Column():
            texto = gr.Textbox(
                label="Sua avaliação",
                placeholder="Ex.: A comida estava deliciosa, mas o atendimento demorou...",
                lines=7,
            )
            with gr.Row():
                classificar = gr.Button("Classificar", variant="primary")
                gr.ClearButton([texto], value="Limpar")
        with gr.Column():
            resultado = gr.Textbox(
                label="Classificação",
                placeholder="O resultado aparecerá aqui.",
                interactive=False,
                elem_id="resultado",
            )

    gr.Examples(
        examples=[
            ["A comida estava deliciosa e o atendimento foi excelente."],
            ["O serviço foi lento e a comida chegou fria."],
            ["O restaurante é razoável, nada de especial."],
        ],
        inputs=texto,
        label="Experimente um exemplo",
    )
    classificar.click(classificar_avaliacao, inputs=texto, outputs=resultado)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
