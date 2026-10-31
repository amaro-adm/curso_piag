#pip install vosk

#Baixar audios:
#https://samplelib.com/sample-wav.html

#baixar modelo:
#https://alphacephei.com/vosk/models

import os
import json
import wave
from vosk import Model, KaldiRecognizer

def transcrever_audio_simples(arquivo_audio):
    # 1. Caminho onde a pasta do modelo baixado deve estar
    pasta_modelo = "NLP/vosk-model-small-en-us-0.15"
    
    # 2. Carrega o modelo de IA na memória do computador
    print("🤖 Carregando o modelo de linguagem...")
    modelo = Model(pasta_modelo)
    
    # 3. Abre o arquivo de áudio (.wav) para leitura de bytes (rb)
    print("🎵 Abrindo o arquivo de áudio...")
    com_audio = wave.open(arquivo_audio, "rb")
    
    # 4. Descobre a taxa de amostragem (frequência) do áudio automaticamente
    taxa_amostragem = com_audio.getframerate()
    
    # 5. Configura o reconhecedor de voz com o modelo e a frequência do áudio
    reconhecedor = KaldiRecognizer(modelo, taxa_amostragem)
    
    texto_completo = ""

    print("📝 Transcrevendo áudio em pedaços...")
    # 6. Loop para ler o áudio aos poucos (em blocos de som)
    while True:
        # Lê um pedaço de 4000 frames do arquivo de áudio
        bloco_som = com_audio.readframes(4000)
        
        # Se o bloco estiver vazio, significa que o áudio acabou
        if len(bloco_som) == 0:
            break
        
        # O Vosk processa esse pedaço. Se ele identificar uma frase completa, retorna True
        if reconhecedor.AcceptWaveform(bloco_som):
            # O resultado vem como texto formatado em JSON, então convertemos para dicionário Python
            resultado_json = json.loads(reconhecedor.Result())
            # Pegamos apenas o texto transcrito e adicionamos ao nosso resultado
            texto_completo += resultado_json.get("text", "") + " "

    # 7. Pega o último pedaço de áudio que sobrou no final do arquivo
    resultado_final_json = json.loads(reconhecedor.FinalResult())
    texto_completo += resultado_final_json.get("text", "")

    # 8. Fecha o arquivo de áudio liberando a memória
    com_audio.close()
    
    return texto_completo.strip()


# --- PONTO DE ENTRADA DO SCRIPT ---
if __name__ == "__main__":
    # Coloque aqui o nome do seu arquivo de áudio (lembre-se: formato .wav, mono)
    meu_arquivo = "NLP/sample-speech-1m.wav"
    
    # Executa a nossa função e exibe o resultado na tela
    texto_convertido = transcrever_audio_simples(meu_arquivo)
    
    print("\n✨ --- RESULTADO DA TRANSCRIÇÃO --- ✨")
    print(texto_convertido)