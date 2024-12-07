import cv2
import numpy as np

def contour_centroid(contour):
    # Calcular el centroide de un contorno
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    return None

def compare_contours(contours1, contours2, area_threshold=1000, centroid_threshold=30):
    # Comparar los centroides y el área de los contornos de dos frames
    centroids1 = [contour_centroid(cnt) for cnt in contours1]
    centroids2 = [contour_centroid(cnt) for cnt in contours2]
    
    # Si no hay centroides en alguno de los frames, consideramos que no hay movimiento
    if not centroids1 or not centroids2:
        return False
    
    # Comparar la diferencia en los centroides
    diff_centroids = sum(np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(centroids1, centroids2))
    
    # Comparar la diferencia en el área de los contornos
    area1 = sum(cv2.contourArea(cnt) for cnt in contours1)
    area2 = sum(cv2.contourArea(cnt) for cnt in contours2)
    
    # Si la diferencia en área es significativa o los centroides se desplazan demasiado, consideramos que hay movimiento
    if abs(area1 - area2) > area_threshold or diff_centroids > centroid_threshold:
        return False
    
    return True

def find_contour_frames(video_path, min_contours=5, movement_threshold=100, area_threshold=1000, quiet_frames_threshold=5):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return None, None, None, None
    
    start_frame = None
    quiet_frame = None
    end_frame = None
    frame_number = 0
    prev_contours = None
    quiet_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir la imagen de BGR a HSV
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Definir los rangos para el color rojo
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        # Crear las máscaras
        mask1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(image_hsv, lower_red_2, upper_red_2)

        # Combinar las máscaras
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Detectar contornos
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > 60 and cv2.boundingRect(cnt)[3] > 40]
        
        num_contours = len(contours)
        
        # Verificar si ya se encontraron 5 contornos
        if num_contours >= min_contours:
            if start_frame is None:
                start_frame = frame_number  # Primer frame con 5 contornos

            # Verificar si los contornos están quietos comparando con el frame anterior
            if prev_contours is not None and compare_contours(prev_contours, contours, area_threshold, movement_threshold):
                quiet_count += 1
                if quiet_count >= quiet_frames_threshold and quiet_frame is None:
                    quiet_frame = frame_number  # Primer frame con dados quietos
        
        # Terminar cuando ya no haya al menos 5 contornos
        if num_contours < min_contours and start_frame is not None:
            end_frame = frame_number
            break
        
        prev_contours = contours  # Actualizar contornos previos
        frame_number += 1

    cap.release()
    
    return start_frame, quiet_frame, end_frame

video_path = "tirada_1.mp4" 
start, quiet, end = find_contour_frames(video_path, min_contours=5, movement_threshold=3, area_threshold=1000, quiet_frames_threshold=1)

if start is not None and quiet is not None and end is not None:
    print(f"El primer frame con 5 dados es: {start}")
    print(f"El primer frame con 5 dados quietos es: {quiet}")
    print(f"El último frame con 5 dados es: {end}")
else:
    print("No se encontraron 5 dados en el video.")
