import cv2
import numpy as np
import os

# Obtener la ruta absoluta de los archivos YOLO
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'

# Cargar el modelo YOLOv3
net = cv2.dnn.readNet(weights_path, config_path)

# Cargar las clases de objetos
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Iniciar la cámara
cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # Preprocesamiento de la imagen para YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Obtener las capas de salida
    # Obtener las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)



    # Listas para almacenar cajas delimitadoras, confianzas y clases
    boxes = []
    confidences = []
    class_ids = []

    # Procesar las detecciones
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supresión no máxima para eliminar detecciones superpuestas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Mostrar las detecciones en la imagen
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label} {confidence}', (x, y - 5), font, 1, color, 1)

    cv2.imshow('YOLO Object Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()