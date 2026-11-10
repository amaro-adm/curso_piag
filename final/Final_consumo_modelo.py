import streamlit as st
import numpy as np
import tensorflow as tf
import os
import plotly.graph_objects as go
import pandas as pd

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Aço Forte S.A. - Qualidade",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Aço Forte S.A. - Sistema de Diagnóstico de Peças")
st.markdown("Sistema de inferência baseada em Rede Neural Artificial para previsão de defeitos na linha de produção.")
st.divider()

# =====================================================================
# VARIÁVEIS GLOBAIS E FUNÇÕES DE CACHE
# =====================================================================
# Cache para carregar o modelo apenas uma vez, melhorando a performance
@st.cache_resource
def carregar_modelo_e_parametros(pasta_modelo):
    prefixo = os.path.join(pasta_modelo, "modelo_treinado")
    
    # 1. Tentar carregar arquivos de normalização
    try:
        X_mean = np.load(f"{prefixo}_mean.npy")
        X_std = np.load(f"{prefixo}_std.npy")
    except FileNotFoundError:
        st.error(f"Arquivos .npy não encontrados no caminho: {pasta_modelo}")
        return None, None, None, None, None, None, None, None
        
    # 2. Recriar arquitetura (Mesma do treinamento)
    W1 = tf.Variable(tf.zeros([5, 64]))
    b1 = tf.Variable(tf.zeros([64]))
    W2 = tf.Variable(tf.zeros([64, 32]))
    b2 = tf.Variable(tf.zeros([32]))
    W_out = tf.Variable(tf.zeros([32, 1]))
    b_out = tf.Variable(tf.zeros([1]))

    # 3. Restaurar Pesos
    checkpoint = tf.train.Checkpoint(W1=W1, b1=b1, W2=W2, b2=b2, W_out=W_out, b_out=b_out)
    
    # expect_partial() evita warnings caso nem todas as variáveis do checkpoint sejam usadas
    status = checkpoint.restore(f"{prefixo}-1").expect_partial()
    
    return X_mean, X_std, W1, b1, W2, b2, W_out, b_out

def forward(X, W1, b1, W2, b2, W_out, b_out):
    z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)
    return tf.matmul(z2, W_out) + b_out

# =====================================================================
# CÁLCULO DE IMPORTÂNCIA DE FEATURES (SALIENCY MAP SIMPLIFICADO)
# =====================================================================
# Como usamos uma rede neural não-linear (com ativações ReLU e múltiplas camadas), 
# os pesos sozinhos (ex: W1) não dizem qual feature é a mais importante globalmente.
# Usamos Gradientes (Saliency) em relação à entrada para ver o que mais afetou a decisão DAQUELA peça específica.
def calcular_importancia_gradiente(X_tensor, y_pred, W1, b1, W2, b2, W_out, b_out):
    with tf.GradientTape() as tape:
        tape.watch(X_tensor) # Monitora os dados de entrada
        logits = forward(X_tensor, W1, b1, W2, b2, W_out, b_out)
        probabilidade = tf.nn.sigmoid(logits)
    
    # Calcula a derivada da probabilidade final em relação a CADA uma das 5 entradas
    gradientes = tape.gradient(probabilidade, X_tensor)
    
    # Pega o valor absoluto (importa a magnitude do impacto, seja positivo ou negativo)
    importancia = np.abs(gradientes.numpy()[0])
    
    # Normaliza para porcentagem (soma = 100%)
    if np.sum(importancia) > 0:
         importancia_pct = (importancia / np.sum(importancia)) * 100
    else:
         importancia_pct = np.zeros(5)
         
    return importancia_pct

# =====================================================================
# INTERFACE LATERAL (SIDEBAR) - CONFIGURAÇÃO DO DIRETÓRIO
# =====================================================================
st.sidebar.header("📁 Configuração do Modelo")
st.sidebar.markdown("Informe o caminho absoluto para a pasta onde os 5 arquivos estão salvos.")

pasta_modelo_padrao = r"C:\\caminho\\para\\sua\\pasta\\modelo"
pasta_modelo_input = st.sidebar.text_input("Caminho do Diretório:", value=pasta_modelo_padrao)

# Carrega os dados se o caminho não for o padrão
modelo_carregado = False
if pasta_modelo_input != pasta_modelo_padrao and os.path.exists(pasta_modelo_input):
    X_mean, X_std, W1, b1, W2, b2, W_out, b_out = carregar_modelo_e_parametros(pasta_modelo_input)
    if X_mean is not None:
         st.sidebar.success("Modelo e Parâmetros de Escala carregados com sucesso!")
         modelo_carregado = True
elif pasta_modelo_input != pasta_modelo_padrao and not os.path.exists(pasta_modelo_input):
    st.sidebar.error("O diretório informado não existe.")
else:
    st.sidebar.info("Aguardando configuração do diretório...")

# Nomes das colunas na ordem exata de entrada
nomes_features = [
    "Diâmetro (mm)",
    "Espessura (mm)",
    "Temp. Têmpera (°C)",
    "Vibração (Hz)",
    "Rugosidade (Ra)"
]

# =====================================================================
# INTERFACE PRINCIPAL - ENTRADA DE DADOS E PREDIÇÃO
# =====================================================================
if not modelo_carregado:
    st.warning("👈 Por favor, informe o caminho correto da pasta com os arquivos do modelo na barra lateral para iniciar o diagnóstico.")
else:
    st.subheader("Entrada de Dados do Sensor")
    st.markdown("Insira as medições da nova peça produzida para obter o diagnóstico.")
    
    # Layout em colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        v_diametro = st.number_input("Diâmetro (mm)", value=47.00, step=0.01, format="%.2f")
        v_espessura = st.number_input("Espessura (mm)", value=14.00, step=0.01, format="%.2f")
    
    with col2:
        v_temp = st.number_input("Temp. Têmpera (°C)", value=830.00, step=1.00, format="%.1f")
        v_vibracao = st.number_input("Vibração (Hz)", value=10.00, step=0.10, format="%.1f")
        
    with col3:
         v_rugosidade = st.number_input("Rugosidade (Ra)", value=1.20, step=0.01, format="%.2f")

    # Botão de Ação
    if st.button("🔍 Avaliar Peça", type="primary"):
        # 1. Preparar o array de entrada
        nova_peca_raw = np.array([[v_diametro, v_espessura, v_temp, v_vibracao, v_rugosidade]], dtype=np.float32)
        
        # 2. Aplicar Normalização (Z-Score) exatamente como no treino
        nova_peca_scaled = (nova_peca_raw - X_mean) / X_std
        
        # Converter para tensor
        X_tensor = tf.constant(nova_peca_scaled, dtype=tf.float32)
        
        # 3. Inferência (Forward Pass)
        logits = forward(X_tensor, W1, b1, W2, b2, W_out, b_out)
        prob_tensor = tf.nn.sigmoid(logits)
        probabilidade = prob_tensor.numpy()[0][0]
        
        # 4. Avaliar Importância via Gradiente (Saliency Map local)
        importancia_pct = calcular_importancia_gradiente(X_tensor, prob_tensor, W1, b1, W2, b2, W_out, b_out)
        
        # 5. Exibir Resultados
        st.divider()
        st.subheader("📊 Resultado do Diagnóstico")
        
        r_col1, r_col2 = st.columns([1, 1.5])
        
        with r_col1:
            # Exibe Métricas
            st.metric(label="Probabilidade de Defeito", value=f"{probabilidade * 100:.2f}%")
            
            if probabilidade > 0.5:
                st.error("### 🔴 PEÇA REJEITADA")
                st.markdown("A rede neural detectou padrões indicativos de defeito nesta peça.")
            else:
                st.success("### 🟢 PEÇA APROVADA")
                st.markdown("A peça está dentro dos padrões de conformidade.")
                
            # Mostra dados normalizados (Opcional, para debug/entendimento)
            with st.expander("Ver dados internos (Pós-Normalização)"):
                df_scaled = pd.DataFrame(nova_peca_scaled, columns=nomes_features)
                st.dataframe(df_scaled.style.format("{:.3f}"))
                
        with r_col2:
            st.markdown("#### Impacto de cada variável na decisão atual:")
            st.markdown("*Quais sensores mais influenciaram o resultado desta peça em específico?*")
            
            # Gráfico de Barras Plotly para Importância
            fig = go.Figure(go.Bar(
                x=importancia_pct,
                y=nomes_features,
                orientation='h',
                marker=dict(
                    color=importancia_pct,
                    colorscale='viridis',
                    reversescale=False
                ),
                text=[f"{val:.1f}%" for val in importancia_pct],
                textposition='auto'
            ))
            
            fig.update_layout(
                xaxis_title="Peso de Decisão (%)",
                yaxis_title="",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)