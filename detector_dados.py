import cv2
import numpy as np

def red_mask(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rango de color rojo en HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Crear máscaras para tonos rojos
    mask1 = cv2.inRange(frame_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(frame_hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def detectar_dados(prev_frame, frame, frame_original):

    dados = []
    prev_canny = cv2.Canny(prev_frame, 950, 1500)
    canny = cv2.Canny(frame, 950, 1500)
    prev_frame_canny = cv2.dilate(prev_canny, None, iterations=2)
    frame_canny = cv2.dilate(canny, None, iterations=2)
    
    contours_prev_frame, _ = cv2.findContours(prev_frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_frame, _ = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_prev_dados = []
    contornos_dados = []

    # Filtramos los contornos por su ratio y area
    for contour in contours_prev_frame:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h
        area = w * h
        
        if 0.85 < ratio < 1.15 and 2500 < area < 20000:
            contornos_prev_dados.append((x, y, w, h))

    for contour in contours_frame:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / h
        area = w * h
        if 0.85 < ratio < 1.15 and 2500 < area < 20000:
            contornos_dados.append((x, y, w, h))
    # Si ambas listas tienen 5 elementos comprobamos si estan quietos
    if len(contornos_prev_dados) == 5 and len(contornos_dados) == 5:
        # Comparamos los contornos para comprobar si hubo movimiento
        for cont_prev, cont_actual in zip(contornos_prev_dados, contornos_dados):
            x1, y1, w1, h1 = cont_prev
            x2, y2, w2, h2 = cont_actual

            # Compara la diferencia de las posiciones x e y
            if abs(x1 - x2) > 5 or abs(y1 - y2) > 5:
                return None
        # Si los dados estan quietos contamos el puntaje de cada dado.

        for contorno in contornos_dados:
            x, y, w, h = contorno
            dado = frame_original[y:y+h, x:x+w]
            dado_gris = cv2.cvtColor(dado, cv2.COLOR_BGR2GRAY)
            _, dado_gris = cv2.threshold(dado_gris, 100, 255, cv2.THRESH_BINARY)

            circles = cv2.HoughCircles(dado_gris, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=20, param2=10, minRadius=4, maxRadius=10)
            # Contar los círculos detectados
            if circles is not None:
                dados.append((contorno,len(circles[0])))
        return dados


for i in range(1,5):    
    cap = cv2.VideoCapture(f'./tiradas/tirada_{i}.mp4')  
    largo = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    formato_salida = cv2.VideoWriter_fourcc(*'mp4v')
    video_salida = cv2.VideoWriter(f'./outputs/salida_{i}.mp4', formato_salida, fps, (largo, alto))

    frames = []
    while (cap.isOpened()): 
        ret, frame = cap.read() 

        if ret == True:
            if len(frames) == 0:
                frames.append(frame)
                continue
            
            prev_frame = frames[-1]
            frames.append(frame)
            
            dados_frame = red_mask(frame)
            dados_prev_frame =  red_mask(prev_frame)
            dados = detectar_dados(dados_prev_frame,dados_frame,frame)

            if dados:
                for (contorno, puntaje) in dados:
                    x, y, h, w = contorno
                    # Dibujar el bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)  
                    cv2.putText(frame, f"{puntaje}", (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                    
            video_salida.write(frame)
        else:  
            break  

    cap.release() # Libera el objeto 'cap', cerrando el archivo.
    video_salida.release()