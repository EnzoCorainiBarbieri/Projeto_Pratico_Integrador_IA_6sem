"""
Detector de Emoções em Tempo Real (OpenCV + DeepFace)

Melhorias incluídas:
- Organização em funções
- Processamento a cada N frames e redimensionamento para ganhar performance
- Exibição da emoção dominante com confiança e top-3 emoções
- Desenho do bounding box do rosto (quando disponível)
- Cálculo e exibição de FPS
- Contagem acumulada de emoções e reset via tecla
- Log em CSV com timestamp e emoção, flush imediato
- Salvamento opcional do recorte do rosto em disco
- Tratamento de erros com logging
- Argumentos de linha de comando para personalização

Teclas:
- q: sair
- r: resetar contadores
- s: alternar salvar recortes do rosto

Requisitos:
  pip install opencv-python deepface

Uso:
  python detector_emocoes_optimizado.py --camera 0 --skip 10 --width 640 --height 480 --save-faces 0
"""

import argparse
import csv
import datetime as dt
import logging
import os
from collections import Counter, deque

import cv2
from deepface import DeepFace


# ===================== Configuração de logging =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("emo-detector")


# ===================== Utilidades =====================

def now_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ===================== Núcleo do detector =====================

def inicializar_camera(index: int, width: int | None, height: int | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    # Tenta reduzir latência
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def analisar_emocoes(frame_bgr) -> dict | None:
    """Retorna o resultado bruto do DeepFace para o primeiro rosto.
    Pode retornar None se nada for detectado ou em erro recuperável.
    """
    try:
        result = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",  # simples e rápido
        )
        # A API pode devolver um dict ou lista de dicts dependendo da versão
        if isinstance(result, list):
            result = result[0] if result else None
        return result
    except Exception as e:
        logger.debug(f"DeepFace.analyze falhou: {e}")
        return None


def extrair_bbox(region: dict, frame_shape):
    if not region:
        return None
    try:
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        h_img, w_img = frame_shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 > x and y2 > y:
            return (x, y, x2, y2)
    except Exception:
        pass
    return None


def desenhar_overlay(frame, info: dict, fps: float, counts: Counter):
    """Desenha texto, FPS, top-3 emoções e caixa do rosto."""
    h, w = frame.shape[:2]

    # Caixa do rosto
    bbox = extrair_bbox(info.get("region"), frame.shape) if info else None
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Texto base
    y0 = 30
    dy = 28

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Emoção dominante
    if info and "dominant_emotion" in info and "emotion" in info:
        dom = info["dominant_emotion"]
        conf = info["emotion"].get(dom, 0.0)
        cv2.putText(frame, f"Dominante: {dom} ({conf:.1f})", (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Top-3 emoções
        ranked = sorted(info["emotion"].items(), key=lambda x: x[1], reverse=True)[:3]
        txt = ", ".join([f"{k}:{v:.1f}" for k, v in ranked])
        cv2.putText(frame, f"Top3: {txt}", (10, y0 + 2 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Contadores (uma linha sintetizada)
    if counts:
        top_counts = ", ".join([f"{k}:{v}" for k, v in counts.most_common(3)])
        cv2.putText(frame, f"Contagem: {top_counts}", (10, y0 + 3 * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

    return frame


def salvar_face(frame_bgr, bbox, out_dir, prefix):
    if not bbox:
        return None
    ensure_dir(out_dir)
    x1, y1, x2, y2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    fname = f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, crop)
    return fpath


# ===================== Loop principal =====================

def run(camera_index: int, skip: int, resize_w: int, resize_h: int, save_faces: bool, faces_dir: str,
        csv_path: str):
    cap = inicializar_camera(camera_index, resize_w, resize_h)
    logger.info("Câmera inicializada")

    ensure_dir(os.path.dirname(csv_path) or ".")
    csv_new = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if csv_new:
        csv_writer.writerow(["timestamp", "emotion", "confidence"])  # header
        csv_file.flush()

    counts = Counter()
    frame_id = 0
    last_info = None
    fps_window = deque(maxlen=30)
    save_faces_flag = bool(save_faces)

    logger.info("Pressione 'q' para sair, 'r' para resetar contadores, 's' para alternar salvar faces")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Falha ao ler frame da câmera. Encerrando...")
            break

        frame_id += 1

        # Medição de FPS
        t0 = cv2.getTickCount()

        # Processa a cada N frames
        if frame_id % max(1, skip) == 0:
            # Redimensiona para acelerar a análise (sem alterar o frame original de exibição)
            resized = cv2.resize(frame, (resize_w, resize_h)) if resize_w and resize_h else frame
            info = analisar_emocoes(resized)
            if info and "dominant_emotion" in info and "emotion" in info:
                dom = info["dominant_emotion"]
                conf = float(info["emotion"].get(dom, 0.0))
                counts[dom] += 1
                csv_writer.writerow([now_str(), dom, f"{conf:.3f}"])
                csv_file.flush()

                # Salvar recorte do rosto se habilitado
                if save_faces_flag:
                    bbox = extrair_bbox(info.get("region"), resized.shape)
                    # Ajusta bbox para coordenadas do frame exibido se redimensionado
                    if bbox and resized is not frame:
                        rx = frame.shape[1] / resized.shape[1]
                        ry = frame.shape[0] / resized.shape[0]
                        x1, y1, x2, y2 = bbox
                        bbox = (int(x1 * rx), int(y1 * ry), int(x2 * rx), int(y2 * ry))
                    salvar_face(frame, bbox, faces_dir, prefix=dom)

                last_info = info

        # Atualiza FPS
        t1 = cv2.getTickCount()
        dt_sec = (t1 - t0) / cv2.getTickFrequency()
        fps_window.append(1.0 / dt_sec if dt_sec > 0 else 0.0)
        fps = sum(fps_window) / len(fps_window)

        # Desenha overlay usando a última análise disponível
        overlay_info = last_info if last_info else {"dominant_emotion": "--", "emotion": {}}

        # Ajusta bbox para o tamanho do frame de exibição se last_info veio de imagem redimensionada
        frame = desenhar_overlay(frame, overlay_info, fps, counts)

        cv2.imshow("Detector de Emoções", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counts.clear()
            logger.info("Contadores resetados")
        elif key == ord("s"):
            save_faces_flag = not save_faces_flag
            logger.info(f"Salvar faces: {'ON' if save_faces_flag else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    logger.info("Encerrado com sucesso.")


# ===================== CLI =====================

def parse_args():
    p = argparse.ArgumentParser(description="Detector de Emoções em Tempo Real")
    p.add_argument("--camera", type=int, default=0, help="Índice da câmera (padrão: 0)")
    p.add_argument("--skip", type=int, default=10, help="Analisar a cada N frames (padrão: 10)")
    p.add_argument("--width", type=int, default=640, help="Largura para análise (padrão: 640)")
    p.add_argument("--height", type=int, default=480, help="Altura para análise (padrão: 480)")
    p.add_argument("--save-faces", type=int, default=0, help="Salvar recortes do rosto 0/1 (padrão: 0)")
    p.add_argument("--faces-dir", type=str, default="faces", help="Pasta para salvar rostos")
    p.add_argument("--csv", type=str, default="emocao_log.csv", help="Caminho do CSV de log")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        skip=max(1, args.skip),
        resize_w=max(1, args.width),
        resize_h=max(1, args.height),
        save_faces=bool(args.save_faces),
        faces_dir=args.faces_dir,
        csv_path=args.csv,
    )
