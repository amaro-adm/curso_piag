# pip install opencv-contrib-python==4.12.0.88

import cv2  # Importa a biblioteca OpenCV para processamento de imagem e visão computacional
import os   # Importa a biblioteca OS para interagir com o sistema operacional

# os.makedirs: Cria um diretório no caminho especificado.
# Parâmetro "rostos": Nome da pasta a ser criada.
# Parâmetro exist_ok=True: Evita erros caso a pasta já exista.
os.makedirs("rostos", exist_ok=True)

# cv2.VideoCapture: Inicializa a captura de vídeo.
# Parâmetro 0: Índice da câmera padrão conectada ao computador (webcam).
camera = cv2.VideoCapture(0)

# cv2.CascadeClassifier: Carrega um modelo pré-treinado de classificação em cascata (Haar Cascade).
# Parâmetro cv2.data.haarcascades + "...": Caminho completo para o arquivo XML do classificador de rostos frontais.
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

contador = 0

# Loop principal para capturar os quadros (frames) da câmera continuamente.
while contador < 10:

    # camera.read(): Lê o próximo frame da câmera.
    # Retorna uma tupla: 'ret' (booleano indicando se a leitura deu certo) e 'frame' (a matriz da imagem capturada).
    ret, frame = camera.read()

    # cv2.cvtColor: Converte o espaço de cores de uma imagem.
    # Parâmetro frame: Imagem original colorida.
    # Parâmetro cv2.COLOR_BGR2GRAY: Código de conversão de BGR (Padrão OpenCV) para Tons de Cinza.
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detector.detectMultiScale: Detecta objetos (rostos) na imagem em escala de cinza.
    # Parâmetro cinza: Imagem de entrada convertida em escala de cinza.
    # Parâmetro 1.1 (scaleFactor): O quanto a imagem é reduzida a cada escala de tamanho da busca.
    # Parâmetro 5 (minNeighbors): Número mínimo de vizinhos que cada retângulo candidato deve ter para reter o rosto (filtra falsos positivos).
    rostos = detector.detectMultiScale(cinza, 1.1, 5)

    # Itera sobre cada rosto detectado, retornando as coordenadas (x, y) do canto superior esquerdo, largura (w) e altura (h).
    for (x, y, w, h) in rostos:

        # Recorta a região de interesse (ROI) correspondente apenas ao rosto na imagem em escala de cinza.
        rosto = cinza[y:y+h, x:x+w]

        # cv2.resize: Redimensiona uma imagem.
        # Parâmetro rosto: Matriz da imagem recortada.
        # Parâmetro (200, 200): Nova resolução padrão de largura e altura em pixels.
        rosto = cv2.resize(rosto, (200, 200))

        # cv2.imwrite: Salva uma imagem no disco.
        # Parâmetro f"rostos/{contador}.jpg": Caminho e nome do arquivo gerado de forma dinâmica.
        # Parâmetro rosto: Matriz da imagem que será salva.
        cv2.imwrite(f"rostos/{contador}.jpg", rosto)

        contador += 1

        # cv2.rectangle: Desenha um retângulo na imagem colorida original para visualização em tempo real.
        # Parâmetro frame: Imagem onde o retângulo será desenhado.
        # Parâmetro (x,y): Coordenada inicial (canto superior esquerdo).
        # Parâmetro (x+w,y+h): Coordenada final (canto inferior direito).
        # Parâmetro (0,255,0): Cor da linha em formato BGR (Verde puro).
        # Parâmetro 2: Espessura da linha do retângulo em pixels.
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # cv2.imshow: Exibe uma imagem em uma janela gráfica.
    # Parâmetro "Cadastro": Título da janela.
    # Parâmetro frame: Imagem atualizada com o retângulo desenhado.
    cv2.imshow("Cadastro", frame)

    # cv2.waitKey: Espera por um evento de tecla por um tempo determinado.
    # Parâmetro 100: Tempo de espera em milissegundos (0.1 segundos).
    # ord("q"): Obtém o código ASCII da tecla 'q'. O operador '& 0xFF' garante compatibilidade de arquitetura de 32/64 bits.
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

# camera.release(): Libera a webcam para ser utilizada por outros aplicativos ou pelo sistema.
camera.release()

# cv2.destroyAllWindows: Fecha todas as janelas abertas pelo OpenCV durante a execução.
cv2.destroyAllWindows()