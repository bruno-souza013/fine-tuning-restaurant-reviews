import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/dataset.csv", sep=",", low_memory=False)

df["text_pt"] = df["text_translation_pt"].fillna(df["text"])

df = df[["text_pt", "rating"]].dropna()
df["text_pt"] = df["text_pt"].str.strip()

df = df[df["text_pt"] != ""]


def create_label(rating):
    if rating <= 2.0:
        return "negative"
    if rating <= 3.5:
        return "neutral"
    return "positive"


df["label"] = df["rating"].apply(create_label)

rotulos_por_texto = df.groupby("text_pt")["label"].nunique()
textos_conflitantes = rotulos_por_texto[rotulos_por_texto > 1].index

print("Grupos com rótulos conflitantes:", len(textos_conflitantes))

df = df[~df["text_pt"].isin(textos_conflitantes)]

df = df.drop_duplicates(subset="text_pt")

#======== balanceamento do dataset por label
amostra = (
    df.groupby("label", group_keys=False)
    .sample(n=5000, random_state=13)
    .sample(frac=1, random_state=13)
    .reset_index(drop=True)
)

amostra = amostra[["text_pt", "label"]]

amostra.to_csv(
    "data/dataset_classificacao.csv",
    index=False,
    encoding="utf-8"
    )

print("\nDataset salvo: ")
print(amostra.shape)
print(amostra['label'].value_counts())

#======== separacao entre treino e teste

treino, restante = train_test_split(
    amostra,
    test_size=0.2,
    random_state=13,
    stratify=amostra['label']
)

validacao, teste = train_test_split(
    restante,
    test_size=0.5,
    random_state=13,
    stratify=restante['label']
)

treino.to_csv("data/train.csv", index=False, encoding="utf-8")
validacao.to_csv("data/validation.csv", index=False, encoding="utf-8")
teste.to_csv("data/test.csv", index=False, encoding="utf-8")

print("\nTamanhos:")
print("Treino:", treino.shape)
print("Validação:", validacao.shape)
print("Teste:", teste.shape)

print("\nClasses no teste:")
print(teste["label"].value_counts())