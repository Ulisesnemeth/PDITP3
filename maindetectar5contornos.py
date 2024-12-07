import cv2
import numpy as np
import matplotlib.pyplot as plt

def contour_centroid(contour):
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return (cx, cy)
    return None

def compare_contours(framenumber, contours1, contours2, area_threshold=1000, centroid_threshold=30):
    centroids1 = [contour_centroid(cnt) for cnt in contours1]
    centroids2 = [contour_centroid(cnt) for cnt in contours2]
    
    if not centroids1 or not centroids2:
        return False
    
    diff_centroids = sum(np.linalg.norm(np.array(c1) - np.array(c2)) for c1, c2 in zip(centroids1, centroids2))
    area1 = sum(cv2.contourArea(cnt) for cnt in contours1)
    area2 = sum(cv2.contourArea(cnt) for cnt in contours2)
    
    if abs(area1 - area2) > area_threshold or diff_centroids > centroid_threshold:
        return False
    return True

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def find_contour_frames(video_path, min_contours=5, movement_threshold=100, area_threshold=1000, quiet_frames_threshold=5, tolerance=2, min_area_for_dado=500):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return None, None, None
    
    start_frame = None
    quiet_frame = None
    end_frame = None
    frame_number = 0
    prev_contours = None
    quiet_count = 0
    last_valid_frame = None
    valid_contours_in_a_row = 0
    visible_dados = []  # Guardar los contornos de los dados visibles

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(image_hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(image_hsv, lower_red_2, upper_red_2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] > 60 and cv2.boundingRect(cnt)[3] > 40]

        num_contours = len(contours)

        if num_contours >= min_contours:
            if start_frame is None:
                start_frame = frame_number  # Primer frame con 5 contornos

            last_valid_frame = frame_number  # Actualizamos el último frame válido
            valid_contours_in_a_row += 1

            # Verificar si los contornos están quietos comparando con el frame anterior
            if prev_contours is not None and compare_contours(frame_number, prev_contours, contours, area_threshold, movement_threshold):
                quiet_count += 1
                if quiet_count >= quiet_frames_threshold and quiet_frame is None:
                    print(f"Frame {frame_number}: Hace {quiet_count} frames que los contornos están quietos.")
                    quiet_frame = frame_number  # Primer frame con dados quietos

            # Verificar si alguno de los contornos es más pequeño que el área mínima para un dado
            visible_dados.clear()
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area_for_dado:
                    visible_dados.append(cnt)
            
            if len(visible_dados) < min_contours:
                # Si no hay suficientes dados visibles, considera este el último frame válido
                end_frame = last_valid_frame

        else:
            # Si el número de contornos disminuye, comprobamos si es porque los dados fueron tapados
            if valid_contours_in_a_row > tolerance:
                end_frame = last_valid_frame  # El último frame válido es el último con 5 dados

            valid_contours_in_a_row = 0  # Reiniciar la secuencia de contornos válidos

        prev_contours = contours  # Actualizar contornos previos
        frame_number += 1
    if end_frame is None:
        end_frame = frame_number - 1  # Si no se encontró un último frame con 5 contornos, el último frame es el último del video

    cap.release()
    
    return start_frame, quiet_frame, end_frame

video_path = "tirada_4.mp4" 
start, quiet, end = find_contour_frames(video_path, min_contours=5, movement_threshold=3, area_threshold=1000, quiet_frames_threshold=5, tolerance=2, min_area_for_dado=500)

if quiet is not None and end is not None:
    print(f"El primer frame con 5 dados quietos es: {quiet}")
    print(f"El último frame con 5 dados es: {end}")
else:
    print("No se encontraron 5 dados en el video.")
