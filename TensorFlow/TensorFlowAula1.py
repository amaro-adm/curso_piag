import numpy as np
import tensorflow as tf

# ============================================================
# SIMULAÇÃO DOS DADOS (Para o código rodar perfeitamente)
# ============================================================
# Simulando as 5 variáveis dos sensores (Features)
np.random.seed(42)
X_train = np.random.rand(15000, 5)

# Simulando a coluna de resultado: 0 (aprovado) ou 1 (defeito)
# Criando uma classe "fictícia" do Pandas para simular o .values usado no código
class FakePandasSeries:

    def __init__(self, data):
        self.values = data


y_train = FakePandasSeries(np.random.randint(0, 2, size=(15000, 1)))

# ============================================================
# ETAPA 2: REDE NEURAL COM TENSORFLOW (BAIXO NÍVEL)
# Objetivo: Usar Deep Learning para capturar relações
# não-lineares entre as medições dos sensores da Aço Forte S.A.
# ============================================================

# 1. Converter dados para tensores TensorFlow
# TensorFlow opera com tensores (arrays multidimensionais otimizados para GPU)
X_train_tf = tf.constant(X_train, dtype=tf.float32)
y_train_tf = tf.constant(y_train.values, dtype=tf.float32)

# 2. Definir variáveis treináveis (pesos e bias) manualmente
# Camada 1: Recebe as 5 features dos sensores → Cospe 64 neurônios
W1 = tf.Variable(tf.random.normal([5, 64], stddev=0.1))
b1 = tf.Variable(tf.zeros([64]))

# Camada 2: Recebe 64 neurônios → Cospe 32 neurônios
W2 = tf.Variable(tf.random.normal([64, 32], stddev=0.1))
b2 = tf.Variable(tf.zeros([32]))

# Camada de saída: Recebe 32 neurônios → Cospe 1 (Probabilidade de defeito)
W_out = tf.Variable(tf.random.normal([32, 1], stddev=0.1))
b_out = tf.Variable(tf.zeros([1]))


# 3. Função de forward pass (propagação direta)
def forward(X):
    # Camada Oculta 1: Multiplicação de Matrizes + Bias + Ativação não-linear ReLU
    z1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    # Camada Oculta 2: Multiplicação de Matrizes + Bias + Ativação não-linear ReLU
    z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)

    # Saída bruta (Logits) que será enviada para a função de perda
    output = tf.matmul(z2, W_out) + b_out
    return output


# 4. Função de perda: Binary Cross-Entropy
# Compara os logits previstos com os rótulos reais e calcula o erro médio (Loss)
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    )


# 5. Otimizador Adam — ajusta pesos via gradiente descendente adaptativo
optimizer = tf.optimizers.Adam(learning_rate=0.001)

# 6. Loop de treinamento manual com GradientTape
print("Iniciando o treinamento da Rede Neural de Baixo Nível...\n")

for epoch in range(100):
    # O GradientTape grava as operações matemáticas do forward pass para calcular as derivadas depois
    with tf.GradientTape() as tape:
        predictions = forward(X_train_tf)
        loss = compute_loss(y_train_tf, predictions)

    # Agrupamos todas as variáveis que precisam de ajuste matemático
    var_list = [W1, b1, W2, b2, W_out, b_out]

    # Backpropagation: Calcula os gradientes (derivadas parciais) do erro em relação aos pesos
    grads = tape.gradient(loss, var_list)

    # Atualiza os valores de W e b na memória para reduzir o erro no próximo passo
    optimizer.apply_gradients(zip(grads, var_list))

    # Exibe o progresso a cada 10 épocas
    if epoch % 10 == 0:
        print(f"Época {epoch:02d} -> Erro (Loss): {loss.numpy():.4f}")

print("\nTreinamento concluído com sucesso!")
        

