# Fine-tuning de avaliações de restaurantes com QLoRA

Projeto de estudo de classificação de sentimento em avaliações de
restaurantes escritas em português. O modelo recebe uma avaliação e retorna
uma das três classes:

- `negative`
- `neutral`
- `positive`

O modelo base utilizado é o
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

## Objetivos

- preparar e auditar um dataset de avaliações;
- evitar vazamento entre treino, validação e teste;
- medir o comportamento do modelo base;
- treinar um adapter LoRA usando quantização 4-bit;
- comparar o modelo ajustado com o baseline nas mesmas condições.

## Dataset

O projeto utiliza o
[PT-EN Zomato dataset](https://data.mendeley.com/datasets/448d7kts3k/1),
disponibilizado no Mendeley Data:

> Silva, Marcos; De Olival, Daniel; Branco, Alexandre; Mendonça, Fábio;
> Mostafa, Sheikh Shanawaz; Morgado-Dias, Fernando (2024), “PT-EN Zomato
> dataset”, Mendeley Data, V1, doi:
> [10.17632/448d7kts3k.1](https://doi.org/10.17632/448d7kts3k.1).

O dataset original possui textos, notas e metadados. O pipeline utiliza:

- `text_translation_pt` como texto principal;
- `text` como fallback quando a tradução não está disponível;
- `rating` para derivar o rótulo.

| Nota | Rótulo |
|---|---|
| 1.0 a 2.0 | `negative` |
| 2.5 a 3.5 | `neutral` |
| 4.0 a 5.0 | `positive` |

### Privacidade e versionamento dos dados

Os CSVs não são publicados neste repositório. Eles podem conter informações
potencialmente identificáveis e devem permanecer no ambiente local ou em um
dataset privado do Kaggle. O arquivo `data/creditos.txt` registra a origem do
dataset.

## Auditoria e preparação

O dataset bruto possui 1.192.173 registros. Após o tratamento:

- foram considerados 539.300 textos válidos;
- valores ausentes e textos vazios foram removidos;
- textos foram deduplicados;
- 2 grupos de textos com rótulos conflitantes foram removidos
  completamente;
- foi criada uma amostra balanceada com 5.000 exemplos por classe;
- a divisão final possui 12.000 exemplos de treino, 1.500 de validação e
  1.500 de teste;
- a divisão foi estratificada usando seed `13`;
- não foi encontrada sobreposição textual entre os três conjuntos após
  normalização de espaços, caixa e Unicode.

O conjunto de teste permaneceu reservado para a avaliação final.

## Estrutura do projeto

```text
qlora/
├── data/
│   └── creditos.txt
├── scripts/
│   ├── check_format.py
│   ├── check_model_loading.py
│   ├── evaluate_baseline.py
│   └── explore_dataset.py
├── src/
│   ├── prepare_dataset.py
│   └── treinar_qlora.py
├── .gitignore
├── README.md
└── requirements.txt
```

Os arquivos `data/*.csv`, ambientes virtuais, checkpoints e saídas de
treinamento são ignorados pelo Git.

## Ambiente

O desenvolvimento local foi validado com:

- Python 3.12;
- PyTorch com CUDA;
- GPU NVIDIA RTX 4060 com 8 GB de VRAM.

O treinamento também foi preparado para execução em GPU no Kaggle. Nesse
caso, `train.csv` e `validation.csv` podem ser enviados como um dataset
privado. O `test.csv` deve permanecer reservado até a avaliação final.

## Instalação

Crie e ative um ambiente virtual e instale as dependências:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

O PyTorch pode exigir uma instalação específica para a versão CUDA da
máquina. Evite substituir uma instalação CUDA funcional sem necessidade.

## Execução

### Preparar os dados

Coloque o dataset original em `data/dataset.csv` e execute:

```powershell
python src/prepare_dataset.py
```

Esse comando gera localmente:

- `data/dataset_classificacao.csv`;
- `data/train.csv`;
- `data/validation.csv`;
- `data/test.csv`.

### Verificar o formato conversacional

```powershell
python scripts/check_format.py
```

O script aplica o template de conversa do tokenizer e verifica a estrutura
usada no treinamento.

### Verificar o carregamento quantizado

```powershell
python scripts/check_model_loading.py
```

Esse teste carrega o modelo em 4-bit com NF4 e confirma o uso da GPU.

### Executar o baseline

```powershell
python scripts/evaluate_baseline.py
```

O baseline usa o modelo original, sem fine-tuning, e avalia o conjunto de
teste completo. O script calcula accuracy, precision, recall, F1 por classe e
gera a matriz de confusão em:

```text
outputs/baseline_confusion_matrix.png
```

Respostas que não correspondem exatamente a `negative`, `neutral` ou
`positive` são registradas como `invalid` e contabilizadas separadamente.

### Treinar o adapter QLoRA

O script [`src/treinar_qlora.py`](src/treinar_qlora.py) oferece um smoke test
e o treinamento completo:

```powershell
python src/treinar_qlora.py --smoke-test
```

O smoke test usa até 100 exemplos de treino, 20 de validação e 5 passos.
Depois de validar a configuração:

```powershell
python src/treinar_qlora.py
```

O treinamento completo usa uma época, batch size 1, acumulação de gradiente
4, comprimento máximo de 512 tokens e learning rate `2e-4`. O adapter e o
tokenizer são salvos em:

```text
outputs/qlora-restaurant-reviews/
```

Em uma sessão Kaggle, adapte os caminhos para `/kaggle/input` e salve os
artefatos em `/kaggle/working`.

## Configuração QLoRA

O experimento utiliza:

- quantização 4-bit NF4;
- double quantization;
- compute dtype `float16`;
- LoRA com `r=16`;
- `lora_alpha=32`;
- dropout `0.05`;
- target modules de atenção e MLP;
- seed `13`;
- modelo base congelado, treinando apenas os parâmetros do adapter.

No smoke test validado, 18.464.768 de 1.562.179.072 parâmetros foram
treináveis, aproximadamente 1,18% do modelo.

## Baseline registrado

O modelo base foi avaliado em 1.500 exemplos:

- accuracy: `0,6753`;
- F1 macro: aproximadamente `0,59`;
- F1 de `negative`: `0,81`;
- F1 de `neutral`: `0,18`;
- F1 de `positive`: `0,78`.

O principal problema observado foi a classe `neutral`, frequentemente
confundida com `negative` ou `positive`. O resultado do modelo ajustado ainda
precisa ser medido no mesmo `test.csv` antes de qualquer conclusão sobre
melhoria.

## Limitações

- os rótulos são derivados da nota, que é uma aproximação do sentimento do
  texto;
- a tradução automática pode introduzir ruído;
- a amostra balanceada não representa a distribuição original das notas;
- duplicatas aproximadas e agrupamentos por usuário ou restaurante não foram
  usados como critério adicional de divisão;
- respostas generativas inválidas podem ocorrer no baseline;
- uma única configuração de modelo e LoRA não permite generalizar o
  resultado para outras configurações.

## Próximos passos

1. concluir ou confirmar o treinamento completo do adapter;
2. avaliar o modelo ajustado no `test.csv`;
3. gerar a matriz de confusão e métricas do modelo ajustado;
4. comparar baseline e fine-tuned nas mesmas condições;
5. analisar exemplos de erro;
6. registrar hardware, tempo e resultados reais;
7. adicionar inferência com o adapter e atualizar esta documentação.
