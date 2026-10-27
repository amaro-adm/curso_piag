# 1. Importamos as ferramentas do Scikit-learn
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 2. Nossos dados históricos (exemplo simplificado da Aço Forte)
# Colunas: [diametro_mm, espessura_mm, temp_tempera_C, vibracao_hz]
X_historico = np.array([
    [47.0, 14.0, 850, 10], # Peça Perfeita
    [47.1, 14.0, 890, 15], # Peça Defeituosa (Temperatura e vibração altas)
    [47.0, 14.0, 845, 11]  # Peça Perfeita
])

# Rótulos históricos: 0 = Aprovado, 1 = Defeituoso
y_historico = np.array([0, 1, 0]pyth)

# 3. Criamos o modelo de IA e o treinamos (O momento do aprendizado!)
modelo_inspetor = DecisionTreeClassifier()
modelo_inspetor.fit(X_historico, y_historico)

# 4. A peça nova passou no sensor agora! Vamos prever:
peca_nova = np.array([[47.05, 14.0, 885, 14]]) 
previsao = modelo_inspetor.predict(peca_nova)

print(f"Resultado da Inspeção: {'Defeituosa ❌' if previsao[0] == 1 else 'Aprovada ✅'}")