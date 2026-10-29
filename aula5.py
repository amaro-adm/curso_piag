#Instale com: pip install google-generativeai 

# Passo 1: Importamos a biblioteca oficial do Google para IA
import google.generativeai as genai
import streamlit as st

st.title("Inteface Gemini")

# Passo 2: Autenticação. É o "crachá" que diz ao Google quem está a usar o serviço

#gere sua chave aqui: https://aistudio.google.com/app/api-keys?
key = "AIzaSyDd3V4FqeIP_9IY14WbWhkYqNGu7kEhQrs" # Chave Do Prof Hebert Felix

genai.configure(api_key=key)

# Passo 3: Escolha do modelo. O "gemini-1.5-flash" é o motor ideal: rápido e eficiente para texto
modelo = genai.GenerativeModel("gemini-3-flash-preview") #gemini-1.5-flash, gemini-3.1-flash-lite, gemini-3.5-flash ou gemini-3-flash-preview

# Passo 4: O "Prompt" (a instrução ou ordem que damos para a IA)
 
prompt = st.chat_input("Fale com o Gemini:")
# Passo 5: Enviamos o prompt e guardamos a resposta gerada pelo modelo
if prompt:
    st.write(f"User: {prompt}")
    resposta = modelo.generate_content(prompt)
    st.write(resposta.text)

# Passo 6: Exibimos o texto final na tela para o usuário