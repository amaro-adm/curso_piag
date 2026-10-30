import sklearn as sk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import os

amostra = "dados_amostras.csv"
def returnCorrectOpenMode():
    localdir = os.listdir()
    if amostra in localdir:
        return "a"
    else:
        return "w"

df = pd.read_csv(amostra, sep=";")

x = df[["diametro_mm", "espessura_mm", "temp_tempera_C", "vibracao_hz", "rugosidade_Ra"]]

scaler = sk.StandardScaler()

model = sk.LinearRegression()



