import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Variáveis globais
caminho_arquivo = None
X_train_tf, y_train_tf = None, None
X_mean, X_std = None, None
W1 = W2 = W_out = b1 = b2 = b_out = None
historico_perda = []


def carregar_csv():
    global caminho_arquivo, X_train_tf, y_train_tf, X_mean, X_std
    print("\n[INFO] Abrindo janela para selecionar CSV...")

    # parent=root força a janela a abrir NA FRENTE da interface
    caminho_arquivo = filedialog.askopenfilename(
        parent=root,
        title="Selecione o arquivo CSV",
        filetypes=[("CSV files", "*.csv")],
    )

    if not caminho_arquivo:
        print("[AVISO] Nenhum arquivo CSV foi selecionado.")
        return

    try:
        df = pd.read_csv(caminho_arquivo, sep=";")
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

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X - X_mean) / X_std

        X_train_tf = tf.constant(X_scaled, dtype=tf.float32)
        y_train_tf = tf.constant(y, dtype=tf.float32)

        print(
            f"[SUCESSO] CSV carregado com {X_train_tf.shape[0]} amostras."
        )
        messagebox.showinfo(
            "Sucesso",
            f"Arquivo carregado com sucesso!\n{X_train_tf.shape[0]} linhas.",
            parent=root,
        )

    except Exception as e:
        print(f"[ERRO] Falha ao ler CSV: {e}")
        messagebox.showerror(
            "Erro de Leitura", f"Falha ao ler o CSV:\n{str(e)}", parent=root
        )


def treinar_modelo():
    global W1, W2, W_out, b1, b2, b_out, historico_perda
    print("\n[INFO] Botão Treinar clicado...")

    if X_train_tf is None:
        print("[ERRO] Tentativa de treino sem CSV carregado.")
        messagebox.showerror(
            "Erro", "Carregue o arquivo CSV primeiro!", parent=root
        )
        return

    print("[INFO] Iniciando treinamento de 50 épocas...")

    W1 = tf.Variable(tf.random.normal([5, 64], stddev=0.1))
    b1 = tf.Variable(tf.zeros([64]))
    W2 = tf.Variable(tf.random.normal([64, 32], stddev=0.1))
    b2 = tf.Variable(tf.zeros([32]))
    W_out = tf.Variable(tf.random.normal([32, 1], stddev=0.1))
    b_out = tf.Variable(tf.zeros([1]))

    def forward(X):
        z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
        z2 = tf.nn.relu(tf.matmul(z1, W2) + b2)
        return tf.matmul(z2, W_out) + b_out

    def compute_loss(y_true, y_pred):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
        )

    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    historico_perda = []
    epocas = 50

    for epoch in range(epocas):
        with tf.GradientTape() as tape:
            logits_previstos = forward(X_train_tf)
            loss = compute_loss(y_train_tf, logits_previstos)

        grads = tape.gradient(loss, [W1, b1, W2, b2, W_out, b_out])
        optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2, W_out, b_out]))
        historico_perda.append(loss.numpy())

    print(f"[SUCESSO] Treino concluído. Loss Final: {loss.numpy():.4f}")
    messagebox.showinfo(
        "Treinamento Concluído",
        f"Modelo treinado com sucesso!\nLoss final: {loss.numpy():.4f}",
        parent=root,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(historico_perda, label="Loss")
    plt.title("Curva de Aprendizado")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def salvar_modelo():
    global W1, W2, W_out, b1, b2, b_out, X_mean, X_std
    print("\n[INFO] Botão Salvar Modelo clicado...")

    if W1 is None:
        print("[ERRO] Tentativa de salvar modelo sem ter treinado.")
        messagebox.showerror(
            "Erro",
            "Você precisa treinar o modelo (Passo 2) antes de salvar!",
            parent=root,
        )
        return

    print("[INFO] Abrindo caixa de diálogo para salvar...")

    # Força a janela de 'Salvar Como' a abrir NA FRENTE
    caminho_salvamento = filedialog.asksaveasfilename(
        parent=root,
        title="Salvar modelo treinado",
        defaultextension=".ckpt",
        filetypes=[
            ("TensorFlow Checkpoint", "*.ckpt"),
            ("Todos os arquivos", "*.*"),
        ],
    )

    if not caminho_salvamento:
        print("[AVISO] O usuário cancelou o salvamento.")
        return

    try:
        caminho_limpo = os.path.normpath(caminho_salvamento)
        prefixo = caminho_limpo.replace(".ckpt", "")

        pasta = os.path.dirname(prefixo)
        if pasta:
            os.makedirs(pasta, exist_ok=True)

        print(f"[INFO] Gravando checkpoint em: {prefixo}")
        checkpoint = tf.train.Checkpoint(
            W1=W1, b1=b1, W2=W2, b2=b2, W_out=W_out, b_out=b_out
        )
        checkpoint.save(prefixo)

        # Salva também os arquivos de normalização
        np.save(f"{prefixo}_mean.npy", X_mean)
        np.save(f"{prefixo}_std.npy", X_std)

        print("[SUCESSO] Modelo e parâmetros de escala salvos!")
        messagebox.showinfo(
            "Sucesso",
            f"Modelo salvo com sucesso em:\n{prefixo}",
            parent=root,
        )

    except Exception as e:
        print(f"[ERRO] Falha ao salvar: {e}")
        messagebox.showerror(
            "Erro ao Salvar",
            f"Ocorreu um erro ao gravar o arquivo:\n{str(e)}",
            parent=root,
        )


# ============================================================
# INTERFACE GRÁFICA
# ============================================================
root = tk.Tk()
root.geometry("400x250")
root.title("Treinamento de Rede Neural - Aço Forte S.A.")

btn_carregar = tk.Button(
    root, text="1. Carregar CSV", command=carregar_csv, width=25
)
btn_carregar.pack(pady=10)

btn_treinar = tk.Button(
    root, text="2. Treinar Modelo", command=treinar_modelo, width=25
)
btn_treinar.pack(pady=10)

btn_salvar = tk.Button(
    root, text="3. Salvar Modelo", command=salvar_modelo, width=25
)
btn_salvar.pack(pady=10)

root.mainloop()