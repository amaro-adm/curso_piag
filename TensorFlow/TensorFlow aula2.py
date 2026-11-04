
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# ============================================================
# 1. CARREGAMENTO DOS DADOS REAIS (.CSV)
# ============================================================
print("--- Passo 1: Carregando e tratando o arquivo CSV ---")

# Caminho absoluto para o seu arquivo local de 15.000 amostras
caminho_arquivo = r"TensorFlow/dados_amostras.csv"

# Leitura real do arquivo utilizando o Pandas com o separador correto
df = pd.read_csv(caminho_arquivo, sep=";")

# Separando as 5 Features (X) e o Target/Rótulo (y)
X = df[
    [
        "diametro_mm",
        "espessura_mm",
        "temp_tempera_C",
        "vibracao_hz",
        "rugosidade_Ra",
    ]
].values
y = df["resultado"].values.reshape(-1, 1)

# Normalização dos dados (Z-score normalization)
# Crucial para que a escala de temperatura (~850) não abafe a escala de espessura (~14)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1.0  # Evita divisão por zero
X_scaled = (X - X_mean) / X_std

# Convertendo as matrizes tratadas para Tensores oficiais do TensorFlow
X_train_tf = tf.constant(X_scaled, dtype=tf.float32)
y_train_tf = tf.constant(y, dtype=tf.float32)

print(f"Sucesso! Foram carregadas {X_train_tf.shape[0]} linhas da Aço Forte S.A.")
print(f"Formato dos dados de entrada (Tensores X): {X_train_tf.shape}")
print(f"Formato dos dados de saída (Tensores Y): {y_train_tf.shape}\n")


# ============================================================
# 2. DEFINIÇÃO DA ARQUITETURA DA REDE NEURAL (BAIXO NÍVEL)
# ============================================================
print("--- Passo 2: Inicializando os Parâmetros da Rede (W e b) ---")

# Camada Oculta 1: 5 entradas (sensores) -> 64 neurônios
W1 = tf.Variable(tf.random.normal([5, 64], stddev=0.1))
b1 = tf.Variable(tf.zeros([64]))

# Camada Oculta 2: 64 neurônios -> 32 neurônios
W2 = tf.Variable(tf.random.normal([64, 32], stddev=0.1))
b2 = tf.Variable(tf.zeros([32]))

# Camada de Saída: 32 neurônios -> 1 neurônio (Previsão final de defeito)
W_out = tf.Variable(tf.random.normal([32, 1], stddev=0.1))
b_out = tf.Variable(tf.zeros([1]))


# Fluxo de propagação direta (Forward Pass)
def forward(X):
    z1 = tf.nn.relu(tf.matmul(X, W1) + b1)  # Ativação ReLU na Camada 1
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)  # Ativação ReLU na Camada 2
    logits = tf.matmul(z2, W_out) + b_out  # Saída bruta linear
    return logits


# Função de cálculo do erro (Loss) usando Entropia Cruzada Binária
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    )


# Otimizador Adam com taxa de aprendizado padrão
optimizer = tf.optimizers.Adam(learning_rate=0.01)

# ============================================================
# 3. LOOP DE TREINAMENTO REAL (BACKPROPAGATION)
# ============================================================
print("--- Passo 3: Iniciando o Loop de Treinamento Manual ---")

historico_perda = []
epocas = 50

for epoch in range(epocas):
    # Monitorando as operações matemáticas para o cálculo automático das derivadas
    with tf.GradientTape() as tape:
        logits_previstos = forward(X_train_tf)
        loss = compute_loss(y_train_tf, logits_previstos)

    # Lista contendo todos os pesos e vieses que o TensorFlow precisa ajustar
    var_list = [W1, b1, W2, b2, W_out, b_out]

    # Backpropagation: Calculando os gradientes (derivadas da perda em relação aos pesos)
    grads = tape.gradient(loss, var_list)

    # Adam atualiza os pesos de fato na memória
    optimizer.apply_gradients(zip(grads, var_list))

    # Salvando o valor do erro para o gráfico de linha posterior
    historico_perda.append(loss.numpy())

    # Feedbacks visuais no terminal para os alunos acompanharem a evolução
    if (epoch + 1) % 10 == 0 or epoch == 0:
        # Conversão de logits para probabilidades para medir a acurácia atual
        probabilidades = tf.nn.sigmoid(logits_previstos)
        previsoes_binarias = tf.cast(probabilidades > 0.5, tf.float32)
        acuracia = tf.reduce_mean(
            tf.cast(tf.equal(previsoes_binarias, y_train_tf), tf.float32)
        )
        print(
            f"Época {epoch+1:03d}/{epocas} -> Erro (Loss): {loss.numpy():.4f} | Acurácia: {acuracia.numpy()*100:.2f}%"
        )

print("\n--- Treinamento Concluído com Sucesso! ---")
#---------------------------------------Salvar checkpoint
opcao = input("Deseja salvar o modelo treinado? (sim/não): ").strip().lower()
if opcao == "sim":
    # Criamos um gerenciador de Checkpoint apontando para as nossas variáveis treináveis
    checkpoint = tf.train.Checkpoint(
        W1=W1, b1=b1, W2=W2, b2=b2, W_out=W_out, b_out=b_out
    )

# Definimos o caminho e o nome do arquivo (ex: modelo_aco_forte.ckpt)
    # Mude todas as barras para / e remova o 'r' da frente se quiser
    caminho_salvamento = r"C:\Users\Professor\Downloads\TensorFlow\TensorFlow\modelo_aco_forte.ckpt"

    # Salvamos de fato no disco
    checkpoint.save(caminho_salvamento)
    print(f"Modelo salvo com sucesso em: {caminho_salvamento}")
else:
    print("Modelo não foi salvo. Você pode treinar novamente e escolher salvar depois.")

# ============================================================
# 4. PLOTAGEM DA CURVA DE APRENDIZADO
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(historico_perda, color="#10b981", linewidth=2.5, label="Erro (Loss)")
plt.title("Curva de Aprendizado Real - Redução de Erro (Aço Forte S.A.)")
plt.xlabel("Épocas de Treinamento")
plt.ylabel("Valor do Erro (Loss)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()