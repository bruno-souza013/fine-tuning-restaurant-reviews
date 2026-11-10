# QLoRA para classificação de avaliações de restaurantes

Projeto de estudo para fazer fine-tuning com **QLoRA** de um modelo pequeno da
Hugging Face. O modelo receberá uma avaliação de restaurante e classificará o
sentimento em uma das três categorias:

- `negative`
- `neutral`
- `positive`

O modelo base escolhido é o
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

## Dataset

Este projeto utiliza o dataset
[PT-EN Zomato dataset](https://data.mendeley.com/datasets/448d7kts3k/1),
disponibilizado no Mendeley Data.

> Silva, Marcos; De Olival, Daniel; Branco, Alexandre; Mendonça, Fábio;
> Mostafa, Sheikh Shanawaz; Morgado-Dias, Fernando (2024), “PT-EN Zomato
> dataset”, Mendeley Data, V1, doi: [10.17632/448d7kts3k.1](https://doi.org/10.17632/448d7kts3k.1).

O dataset original possui avaliações, notas e metadados de restaurantes. Para
este projeto, são utilizados somente o texto da avaliação e o rótulo derivado
da nota:

| Nota | Rótulo |
|---|---|
| 1.0 a 2.0 | `negative` |
| 2.5 a 3.5 | `neutral` |
| 4.0 a 5.0 | `positive` |

O dataset original não deve ser enviado ao GitHub. Ele contém dados que podem
ser pessoais ou identificáveis. Mantenha os arquivos de dados privados e
verifique os termos de uso antes de redistribuí-los.

## Estrutura atual

```text
qlora/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   └── creditos.txt
├── baseline_test.py
├── explore_dataset.py
├── load_model_test.py
└── test_format.py
```

Os scripts atuais documentam as etapas exploratórias:

1. exploração e limpeza dos dados;
2. criação das classes;
3. separação em treino, validação e teste;
4. preparação do formato conversacional;
5. carregamento do modelo quantizado;
6. teste do modelo antes do fine-tuning.

## Ambiente

O desenvolvimento local foi preparado com:

- Python 3.12;
- PyTorch com CUDA;
- GPU NVIDIA RTX 4060 com 8 GB de VRAM.

## Próximos passos

- criar o script de treinamento QLoRA;
- avaliar o modelo base no conjunto de teste;
- treinar os adaptadores LoRA;
- comparar o resultado antes e depois do fine-tuning;
- adicionar um script de inferência;
- documentar métricas e exemplos de classificação.

