"""
Protótipo de Detecção de Microexpressões (Python)

Funcionalidades:
- Captura vídeo via OpenCV
- Detecta rostos e landmarks faciais com MediaPipe
- Calcula mudanças rápidas nos landmarks para detectar microexpressões
- Exibe resultados na tela
- Registra log em CSV com timestamp, microexpressão detectada e caminho da imagem do rosto
- Salva automaticamente recortes do rosto sempre que uma microexpressão é detectada

"""

import cv2  # Biblioteca para captura de vídeo e manipulação de imagens
import mediapipe as mp  # Biblioteca para detecção de landmarks faciais
import numpy as np  # Para cálculos matemáticos (diferenças de pontos)
import csv  # Para salvar logs em formato CSV
import datetime as dt  # Para registrar timestamps
import os  # Para manipulação de diretórios e caminhos
from collections import deque  # Para armazenar histórico de landmarks

# ================= Configurações =================
FPS_SMOOTH = 10  # Número de frames para cálculo de FPS médio
LOG_CSV = 'microexp_log.csv'  # Nome do arquivo CSV para log
LANDMARK_HISTORY = 5  # Quantos frames lembrar para analisar microexpressão
MICROEXP_THRESHOLD = 2.0  # Mudança mínima nos landmarks para considerar microexpressão
SAVE_FACES_DIR = 'microexp_faces'  # Pasta onde serão salvos os rostos detectados

# Criar pasta para salvar rostos se não existir
os.makedirs(SAVE_FACES_DIR, exist_ok=True)

# Configuração do MediaPipe Face Mesh para detectar landmarks faciais
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Detecta rostos em vídeo contínuo
    max_num_faces=1,  # Detecta apenas um rosto por vez
    refine_landmarks=True,  # Refinar landmarks para olhos, lábios, etc.
    min_detection_confidence=0.5,  # Confiança mínima para detecção inicial
    min_tracking_confidence=0.5  # Confiança mínima para rastrear landmarks
)

# ================= Configuração do CSV =================
csv_file = open(LOG_CSV, 'a', newline='')  # Abrir arquivo CSV em modo append
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'microexpression_detected', 'face_image'])  # Cabeçalho

# ================= Captura de vídeo =================
cap = cv2.VideoCapture(0)  # Inicializa a câmera padrão
if not cap.isOpened():
    print("Não foi possível abrir a câmera")
    exit()

# Armazena histórico dos últimos LANDMARK_HISTORY frames para detectar microexpressões
landmarks_history = deque(maxlen=LANDMARK_HISTORY)

# Loop principal
while True:
    ret, frame = cap.read()  # Captura um frame da câmera
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB (MediaPipe requer RGB)
    results = face_mesh.process(frame_rgb)  # Detecta landmarks faciais no frame

    microexp_detected = False  # Flag para microexpressão detectada
    face_image_path = ''  # Caminho da imagem do rosto salvo

    if results.multi_face_landmarks:  # Se algum rosto foi detectado
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            # Converte coordenadas normalizadas para pixels do frame
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            landmarks.append((x, y))
        landmarks = np.array(landmarks)

        # Adiciona os landmarks atuais ao histórico
        landmarks_history.append(landmarks)

        # Análise simples: diferença entre frames consecutivos para detectar microexpressões
        if len(landmarks_history) >= 2:
            diff = np.linalg.norm(landmarks_history[-1] - landmarks_history[-2], axis=1)  # Diferença entre cada ponto
            mean_diff = np.mean(diff)  # Média das diferenças
            if mean_diff > MICROEXP_THRESHOLD:
                microexp_detected = True  # Microexpressão detectada

                # Salvar recorte do rosto
                x_min, y_min = landmarks[:,0].min(), landmarks[:,1].min()
                x_max, y_max = landmarks[:,0].max(), landmarks[:,1].max()
                # Adicionar margem para não cortar parte do rosto
                x_min, y_min = max(0, x_min-10), max(0, y_min-10)
                x_max, y_max = min(frame.shape[1], x_max+10), min(frame.shape[0], y_max+10)
                face_crop = frame[y_min:y_max, x_min:x_max]
                timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                face_image_path = os.path.join(SAVE_FACES_DIR, f'microexp_{timestamp}.jpg')
                cv2.imwrite(face_image_path, face_crop)  # Salva a imagem do rosto

        # Desenhar landmarks no frame para visualização
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Mostrar resultado na tela
    status_text = 'Microexpressao detectada!' if microexp_detected else 'Normal'
    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if microexp_detected else (0,255,0), 2)

    cv2.imshow('Microexpressao', frame)  # Exibe o frame com landmarks e status

    # Registrar no CSV
    csv_writer.writerow([dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), int(microexp_detected), face_image_path])

    # Tecla para sair
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Libera recursos
cap.release()
csv_file.close()
cv2.destroyAllWindows()
