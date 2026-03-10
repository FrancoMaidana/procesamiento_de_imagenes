import cv2
import numpy as np
import config as cf
#===========================================
# ESTE SCRIPT NO UTILIZA HOUGH CIRCLES PARA LA DETECCION DE TAPAS 
# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
# Cambia esto a True para usar una foto, o a False para usar el video
MODO_IMAGEN_ESTATICA = False  

RUTA_VIDEO = "C:/Users/Dr.Rockzo/Desktop/tesis-franco/SKOL_1M.MOV"
# Guarda un frame de tu video como imagen para calibrar
RUTA_IMAGEN = "botella_caida_hsv.png" 

#!!!!!!!!!!!!!!!!!!!!!! Polígono de la cinta (opcional, para aislar el fondo)!!!!!!!!!!!!!!!!!!!!!!!!!!!
PUNTOS_CINTA = np.array([[0, 0], [600, 0], [600, 800], [0, 800]], np.int32).reshape((-1, 1, 2))
#PUNTOS_CINTA = np.array([[200, 0], [500, 50], [520, 400], [700, 750], [400, 750], [250, 450]], np.int32).reshape((-1, 1, 2))
# Rango HSV para las Tapas (Ajusta según tu color)
TAPA_BAJO = np.array([0, 100, 100])
TAPA_ALTO = np.array([10, 255, 255])

def nothing(x):
    pass

def aislar_tapas_y_desfase():
    
    # 1. Configurar la fuente (Imagen o Video)
    if MODO_IMAGEN_ESTATICA:
        frame_original = cv2.imread(RUTA_IMAGEN)
        if frame_original is None:
            print(f"Error: No se encontró la imagen en {RUTA_IMAGEN}")
            return
        frame_original = cv2.resize(frame_original, (600, 800))
        cv2.imshow("Imagen Original para Calibracion", frame_original)  # Mostrar la imagen original para referencia
    else:
        cap = cv2.VideoCapture(RUTA_VIDEO)

    # 2. Crear ventana de controles
    cv2.namedWindow("Controles Desfase")
    cv2.resizeWindow("Controles Desfase", 450, 250)
    
    # Trackbars para ajustar el tamaño del borrador
    cv2.createTrackbar("Radio Mascara", "Controles Desfase", 60, 150, nothing)
    cv2.createTrackbar("Area Min Tapa", "Controles Desfase", 50, 500, nothing)
    
    # Trackbars para el Paralaje (Perspectiva)
    cv2.createTrackbar("Punto Cero X", "Controles Desfase", 300, 600, nothing)
    cv2.createTrackbar("Punto Cero Y", "Controles Desfase", 400, 800, nothing)
    # 100 = 0% de desfase. 200 = 100% positivo. 0 = 100% negativo.
    cv2.createTrackbar("Fuerza Desfase X", "Controles Desfase", 100, 200, nothing)
    cv2.createTrackbar("Fuerza Desfase Y", "Controles Desfase", 100, 200, nothing)

    while True:
        # --- LECTURA DEL FRAME ---
        if MODO_IMAGEN_ESTATICA:
            frame = frame_original.copy()
            
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar video
                continue
            frame = cv2.resize(frame, (600, 800))

        frame_viz = frame.copy() # Copia para dibujar las líneas guía
        print("probando importacion de parametros", cf.CONFIG['ESPUMA_BAJO'], cf.CONFIG['ESPUMA_ALTO'])
        # --- LEER TRACKBARS ---
        radio_mask = cv2.getTrackbarPos("Radio Mascara", "Controles Desfase")
        area_min = cv2.getTrackbarPos("Area Min Tapa", "Controles Desfase")
        
        p_cero_x = cv2.getTrackbarPos("Punto Cero X", "Controles Desfase")
        p_cero_y = cv2.getTrackbarPos("Punto Cero Y", "Controles Desfase")
        
        # Matemática para convertir de [0-200] a un factor de [-1.0 a 1.0]
        k_x = (cv2.getTrackbarPos("Fuerza Desfase X", "Controles Desfase") - 100) / 100.0
        k_y = (cv2.getTrackbarPos("Fuerza Desfase Y", "Controles Desfase") - 100) / 100.0

        # --- DIBUJAR REFERENCIA VISUAL ---
        # Cruz celeste marcando el "Punto Cero" (Donde la cámara mira recto)
        cv2.drawMarker(frame_viz, (p_cero_x, p_cero_y), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # --- PROCESAMIENTO ---
        # Aislar cinta
        #mask_fondo = np.zeros(frame.shape[:2], dtype=np.uint8)
        #cv2.fillPoly(mask_fondo, [PUNTOS_CINTA], 255)
        frame_roi=frame.copy()
        #frame_roi = cv2.bitwise_and(frame, frame, mask=mask_fondo)

        # Pasar a HSV para buscar las tapas
        hsv_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
        mask_tapas = cv2.inRange(hsv_roi, TAPA_BAJO, TAPA_ALTO)
        
        # Encontrar contornos de las tapas
        contornos_tapas, _ = cv2.findContours(mask_tapas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contornos_tapas:
            if cv2.contourArea(cnt) > area_min:
                M = cv2.moments(cnt) ##calcula el centroide de la tapa detectada
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # --- MATEMÁTICA DEL DESFASE ---
                    # 1. ¿A qué distancia estamos del Punto Cero?
                    distancia_x = cx - p_cero_x
                    distancia_y = cy - p_cero_y
                    
                    # 2. Aplicar la fuerza de desfase
                    desfase_x = int(distancia_x * k_x)
                    desfase_y = int(distancia_y * k_y)
                    
                    # 3. Calcular la nueva coordenada del círculo
                    nuevo_cx = cx + desfase_x
                    nuevo_cy = cy + desfase_y
                    
                    # (OPCIONAL) Aplicar el círculo negro a la matriz HSV si fueras a seguir con el código
                    cv2.circle(hsv_roi, (nuevo_cx, nuevo_cy), radio_mask, (0, 0, 0), -1)

                    # --- DIBUJOS PARA ENTENDER QUÉ ESTÁ PASANDO ---
                    # 1. Punto blanco: El centro real de la tapa detectada
                    cv2.circle(frame_viz, (cx, cy), 4, (255, 255, 255), -1)
                    
                    # 2. Círculo translúcido (Gris oscuro): Simula el "Cortador de Galletas"
                    # Lo dibujamos en el frame_viz para que veas qué taparía
                    overlay = frame_viz.copy()
                    cv2.circle(overlay, (nuevo_cx, nuevo_cy), radio_mask, (0, 0, 0), -1)
                    # Mezclamos la imagen original con el círculo negro al 50% de transparencia
                    cv2.addWeighted(overlay, 0.6, frame_viz, 0.4, 0, frame_viz)
                    
                    # 3. Línea Verde: Te muestra el vector (dirección y fuerza) del desfase
                    cv2.line(frame_viz, (cx, cy), (nuevo_cx, nuevo_cy), (0, 255, 0), 2)
                    
                    # 4. Punto rojo: El nuevo centro desfasado
                    cv2.circle(frame_viz, (nuevo_cx, nuevo_cy), 4, (0, 0, 255), -1)

        # Dibujar los bordes de la cinta para referencia
        cv2.polylines(frame_viz, [PUNTOS_CINTA], True, (255, 0, 0), 2)
        cv2.namedWindow("Calibracion de Desfase (Tapas)")
        cv2.resizeWindow("Calibracion de Desfase (Tapas)", 600, 800)
        cv2.imshow("Calibracion de Desfase (Tapas)", frame_viz)

        # Control de velocidad y salida
        delay = 30 if not MODO_IMAGEN_ESTATICA else 100
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    if not MODO_IMAGEN_ESTATICA:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    aislar_tapas_y_desfase()