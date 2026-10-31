import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os

amostra = "dados_amostras.csv"

df = pd.read_csv(amostra, sep=";")

x = df[["diametro_mm", "espessura_mm", "temp_tempera_C", "vibracao_hz", "rugosidade_Ra"]]

y = df[["resultados"]]

scaler = sk.StandardScaler()

x_scaled, y_scaled = scaler(x, y)

model = sk.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled)



