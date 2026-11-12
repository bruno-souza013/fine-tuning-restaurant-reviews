# Restaurant Review QLoRA

Fine-tuning do [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
para classificar avaliações de restaurantes em português:

```text
negative · neutral · positive
```

<p align="center">
  <a href="https://huggingface.co/spaces/brunoSouza013/fine-tuning-restaurant-reviews">
    <img src="https://img.shields.io/badge/_TESTAR_AVALIADOR_ONLINE-Hugging_Face-yellow?style=for-the-badge&logo=huggingface" />
  </a>
</p>

## Resultado

Avaliação em 1.500 exemplos balanceados:

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Qwen base | 67,53% | ~0,59 |
| Qwen + QLoRA | **81,20%** | **0,81** |

O adapter QLoRA melhorou a accuracy em **13,67 pontos percentuais**. A classe
mais difícil continua sendo `neutral` (F1 `0,72`).

<details>
<summary>Ver métricas por classe</summary>

| Classe | Precision | Recall | F1 |
|---|---:|---:|---:|
| `negative` | 0,86 | 0,89 | 0,87 |
| `neutral` | 0,72 | 0,73 | 0,72 |
| `positive` | 0,86 | 0,82 | 0,84 |

</details>

## Dataset

Fonte: [PT-EN Zomato dataset](https://data.mendeley.com/datasets/448d7kts3k/1).

> Silva, Marcos; De Olival, Daniel; Branco, Alexandre; Mendonça, Fábio;
> Mostafa, Sheikh Shanawaz; Morgado-Dias, Fernando (2024), “PT-EN Zomato
> dataset”, Mendeley Data, V1, doi:
> [10.17632/448d7kts3k.1](https://doi.org/10.17632/448d7kts3k.1).

Os rótulos foram derivados de `rating`:

| Nota | Rótulo |
|---|---|
| 1,0–2,0 | `negative` |
| 2,5–3,5 | `neutral` |
| 4,0–5,0 | `positive` |

O dataset original e os CSVs não são versionados. Mantenha-os localmente ou
em um dataset privado do Kaggle e verifique a licença antes de redistribuí-los.

## Comece aqui

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pipeline:

```powershell
python pipeline.py
```

No menu, escolha a etapa desejada. O treinamento oferece primeiro um smoke
test curto para validar a configuração. Para treinar no Kaggle, envie o código
para o GitHub e use `train.csv` e `validation.csv` como dataset privado.
Preserve `test.csv` para a avaliação final.

## Estrutura

```text
├── pipeline.py  # menu principal
├── data/       # créditos; CSVs locais ignorados pelo Git
├── src/        # preparação dos dados e treinamento
├── scripts/    # checagens, avaliação e inferência
├── outputs/    # resultados locais ignorados pelo Git
└── README.md
```

<details>
<summary>Configuração do experimento</summary>

- QLoRA com quantização 4-bit NF4 e double quantization;
- compute dtype `float16`;
- LoRA `r=16`, `alpha=32`, dropout `0,05`;
- 15.000 exemplos balanceados;
- 12.000 treino, 1.500 validação e 1.500 teste;
- 1 época, comprimento máximo de 512 tokens;
- seed `13`;
- GPU usada no desenvolvimento: NVIDIA RTX 4060 com 8 GB.

</details>

<details>
<summary>Limitações</summary>

- os rótulos são aproximações baseadas na nota, não anotações manuais;
- traduções podem introduzir ruído;
- a amostra balanceada não representa a distribuição original;
- `neutral` é semanticamente ambígua e concentra mais erros;
- o adapter e os dados permanecem fora deste repositório;
- a interface usa um guardrail com o próprio modelo para filtrar textos fora
  do domínio antes da classificação, o que adiciona uma segunda geração por
  consulta e pode produzir falsos positivos ou falsos negativos.

</details>

## Relatórios locais

Os detalhes completos ficam nos arquivos gerados localmente:

- `outputs/result.md` — modelo base;
- `qlora-restaurant-reviews/result.md` — modelo após QLoRA.
