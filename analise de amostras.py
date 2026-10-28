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
espessuraInput = input("Espessura: ")
diametroInput = input("Diâmetro: ")
tempInput = input("Temperatura de têmpera: ")
hzInput = input("Vibração em Hz: ")
raInput = input("Rugosidade: ")

peca_nova = { "diametro_mm": diametroInput,
             "espessura_mm": espessuraInput,
             "temp_tempera_C": tempInput,
             "vibracao_hz": hzInput,
             "rugosidade_Ra": raInput }  # precisa ter todas as colunas

previsao = modelo_inspetor.predict(peca_nova)

print(f"Resultado da Inspeção: {'Defeituosa ❌' if previsao[0] == 1 else 'Aprovada ✅'}")
