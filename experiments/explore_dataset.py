#%%
import pandas as pd
df = pd.read_csv("data/dataset.csv", sep=",", low_memory=False)

# %%
print("Dimensoes: ", df.shape)
print("\nColunas")
print(df.columns.to_list())

print("\nDistribuicao das avaliacoes:")
print(df["rating"].value_counts(dropna=False).sort_index())

print("\nPrimeiras avaliacoes:")
print(df[["text", "text_translation_pt", "rating"]].head())
# %%
#Filtrar textos vazios
texto = df["text_translation_pt"].fillna(df["text"]) 

print("\nTextos vazios:")
print(texto.str.strip().eq("").sum())

print("\nValores ausentes:")
print(df[["text", "text_translation_pt", "rating"]].isna().sum())

print("\nIdiomas:")
print(df["text_language"].value_counts(dropna=False).head(20))

print("\nDuplicatas de texto:")
print(texto.duplicated().sum())

print("\nTamanho dos textos:")
print(texto.dropna().str.len().describe())
# %%

texto_valido = texto.dropna().loc[lambda coluna: coluna.str.strip() != ""]

print("\nTextos validos:", len(texto_valido))
print("Textos válidos unicos:", texto_valido.nunique())
print("Duplicatas entre textos validos:", texto_valido.duplicated().sum())

print("\nAvaliacoes dos textos validos:")
print(
    df.loc[texto_valido.index, "rating"]
    .value_counts()
    .sort_index()
)
# %%
