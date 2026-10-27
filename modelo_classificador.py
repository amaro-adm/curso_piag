import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler

dataFrame = pd.read_csv()

# Separa características dos rolamentos do resultado
X = dataFrame[['diametro_mm', 'espessura_mm', 'temp_tempera_C', 'vibracao_hz, rugosidade_Ra']]
Y = dataFrame['resultado']

scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)

