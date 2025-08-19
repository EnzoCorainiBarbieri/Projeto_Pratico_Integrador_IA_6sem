
import cv2 #biblioteca OpenCV, que é a principal ferramenta para processamento de imagens e vídeo em Python.

# Carrega o classificador pré-treinado para detecção de rosto
#Esse classificador consegue reconhecer padrões típicos de rostos (olhos, nariz, boca)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0) #Inicializa a captura de vídeo, o número 0 indica a câmera padrão do notebook.

#Verifica se a câmera foi aberta corretamente.
if not cap.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

# Loop para capturar frames da câmera
"""
cap.read() lê um frame por vez da câmera.

ret é um booleano que indica se a captura foi bem-sucedida.

frame é a imagem capturada.

Se não conseguir ler, o loop termina.
"""
while True:
    ret, frame = cap.read() 
    if not ret:
        print("Não foi possível ler o frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza, Haar Cascade funciona melhor em imagens monocromáticas.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Desenha retângulos nos rostos detectados
    """
    Para cada rosto detectado, desenha um retângulo verde na imagem.

    (x, y) é o canto superior esquerdo do rosto, w e h são largura e altura.

    (0, 255, 0) é a cor verde no formato BGR.

    2 é a espessura da borda do retângulo.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Camera', frame) # Exibe o frame em uma janela chamada "Camera"

    if cv2.waitKey(1) & 0xFF == ord('q'): # Sai do loop se a tecla 'q' for pressionada
        break

    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1: #Verifica se a janela foi fechada manualmente.
        break

# Libera a captura de vídeo e fecha todas as janelas abertas
"""
cap.release() libera a câmera, permitindo que outro programa a use.

cv2.destroyAllWindows() fecha todas as janelas abertas pelo OpenCV.

Isso garante que a aplicação pare completamente.
"""
cap.release()
cv2.destroyAllWindows()

