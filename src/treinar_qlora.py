import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 13


def formatar_exemplo(exemplo: dict, tokenizer: AutoTokenizer) -> dict:
    mensagens = [
        {
            "role": "system",
            "content": (
                "Classifique avaliações de restaurantes usando apenas: "
                "negative, neutral ou positive."
            ),
        },
        {"role": "user", "content": exemplo["text_pt"]},
        {"role": "assistant", "content": exemplo["label"]},
    ]

    return {
        "text": tokenizer.apply_chat_template(
            mensagens,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def carregar_dataset(caminho: str, tokenizer: AutoTokenizer) -> Dataset:
    dataframe = pd.read_csv(caminho)
    dataset = Dataset.from_pandas(dataframe[["text_pt", "label"]])
    return dataset.map(
        lambda exemplo: formatar_exemplo(exemplo, tokenizer),
        remove_columns=["text_pt", "label"],
    )


def criar_modelo(tokenizer: AutoTokenizer):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    tokenizer.pad_token = tokenizer.eos_token
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true") #teste curto
    parser.add_argument("--output-dir", default="outputs/qlora-restaurant-reviews")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_dataset = carregar_dataset("data/train.csv", tokenizer)
    eval_dataset = carregar_dataset("data/validation.csv", tokenizer)

    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(20, len(eval_dataset))))

    model = criar_modelo(tokenizer)
    output_dir = Path(args.output_dir)

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "logging_steps": 1 if args.smoke_test else 50,
        "eval_strategy": "steps",
        "eval_steps": 5 if args.smoke_test else 500,
        "save_strategy": "no" if args.smoke_test else "epoch",
        "fp16": False,
        "bf16": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "dataset_text_field": "text",
        "max_length": 512,
        "seed": SEED,
    }

    if args.smoke_test:
        training_kwargs["max_steps"] = 5
    else:
        training_kwargs["num_train_epochs"] = 1

    training_args = SFTConfig(**training_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    resultado = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(resultado)
    print(f"Adapter salvo em: {output_dir}")


if __name__ == "__main__":
    main()
