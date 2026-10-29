# 1. Importamos as ferramentas
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# 2. Lendo os dados históricos do CSV
# O arquivo deve ter colunas: diametro_mm;espessura_mm;temp_tempera_C;vibracao_hz;rugosidade_Ra;resultado
df = pd.read_csv("dados_amostras.csv", sep=";")

# 3. Separando variáveis de entrada (X) e rótulo (y)
X_historico = df[["diametro_mm", "espessura_mm", "temp_tempera_C", "vibracao_hz", "rugosidade_Ra"]]
y_historico = df["resultado"]

# 4. Criamos o modelo e treinamos
modelo_inspetor = DecisionTreeClassifier()
modelo_inspetor.fit(X_historico, y_historico)

# 5. Previsão para uma peça nova

peca_nova = { "diametro_mm": float,
             "espessura_mm": float,
             "temp_tempera_C": int,
             "vibracao_hz": float,
             "rugosidade_Ra": int }  # precisa ter todas as colunas

def checkKeyLabel(key):
    keywordArray = [["diametro_mm", "espessura_mm", "temp_tempera_C", "vibracao_hz", "rugosidade_Ra"]]
    labelArray = [["Diâmetro (mm)", "Espessura (mm)", "Temperatura de Têmpera (C°)", "Vibração (Hz)", "Rugosidade (Ra)"]]
    for index in key:
        if labelArray.index() == index.index():
            return labelArray[index]

for key in peca_nova:
    labelKeys = checkKeyLabel(key)
    keys = [].append(input(f"Digite {key}: \n"))
    peca_nova = keys

previsao = modelo_inspetor.predict(peca_nova)

print(f"Resultado da Inspeção: {'Defeituosa ❌' if previsao[0] == 1 else 'Aprovada ✅'}")
