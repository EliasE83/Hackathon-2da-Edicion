import cv2
import numpy as np
import pyttsx3

# Obtener la ruta absoluta de los archivos YOLO
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'

detected_persons = [] 

# Cargar el modelo YOLOv3
net = cv2.dnn.readNet(weights_path, config_path)

# Cargar las clases de objetos
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Iniciar la cámara
cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    height, width, _ = img.shape

    # Preprocesamiento de la imagen para YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

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
    
    # Después de obtener las coordenadas de los objetos
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]

        # Calcular la distancia (esto es solo un ejemplo, necesitarás ajustar según tu calibración)
        focal_length = 300  # Ajusta según tus datos de calibración
        real_height = 1.0  # Altura real del objeto
        pixel_height = h  # Altura del objeto en píxeles
        distance = (focal_length * real_height) / pixel_height

        # Mostrar etiqueta con distancia
        label_with_distance = f'{label} esta a una distancia de {round(distance, 2)} metros'
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label_with_distance, (x, y - 5), font, 1, color, 1)

        
        
        if label == 'botella':
            person_id = str(class_ids[i]) + '_' + str(boxes[i])
            
            if person_id not in detected_persons:

                engine = pyttsx3.init()
                print(label_with_distance)
                engine.say(label_with_distance)
                engine.runAndWait()

                detected_persons.append(person_id)
        

        if label == 'senal de alto':
            person_id = str(class_ids[i]) + '_' + str(boxes[i])
            
            if person_id not in detected_persons:

                engine = pyttsx3.init()
                print(label_with_distance)
                engine.say('Precaucion, hay una ' + label_with_distance)
                engine.runAndWait()

                detected_persons.append(person_id)

        if label == 'semaforo':
            person_id = str(class_ids[i]) + '_' + str(boxes[i])
            
            if person_id not in detected_persons:

                engine = pyttsx3.init()
                print(label_with_distance)
                engine.say('Precaucion, te acercas a un ' + label_with_distance)
                engine.runAndWait()

                detected_persons.append(person_id)

        if label == 'silla':
            if distance < 1.0: 
                person_id = str(class_ids[i]) + '_' + str(boxes[i])
                
                if person_id not in detected_persons:
    
                    engine = pyttsx3.init()
                    print('Precaucion, una ' + label_with_distance)
                    engine.say('Precaucion, una ' +  label_with_distance)
                    engine.runAndWait()

                    detected_persons.append(person_id)
            
        
                detected_persons.append(person_id)    
                detected_persons.append(person_id)        

        if label == 'senal de alto':
            person_id = str(class_ids[i]) + '_' + str(boxes[i])
            
            if person_id not in detected_persons:

                engine = pyttsx3.init()
                print(label_with_distance)
                engine.say('Precaucion, hay una ' + label_with_distance)
                engine.runAndWait()

                detected_persons.append(person_id)

        if label == 'semaforo':
            person_id = str(class_ids[i]) + '_' + str(boxes[i])
            
            if person_id not in detected_persons:

                engine = pyttsx3.init()
                print(label_with_distance)
                engine.say('Precaucion, te acercas a un ' + label_with_distance)
                engine.runAndWait()

                detected_persons.append(person_id)

        if label == 'silla':
            if distance < 1.0: 
                person_id = str(class_ids[i]) + '_' + str(boxes[i])
                
                if person_id not in detected_persons:
    
                    engine = pyttsx3.init()
                    print('Precaucion, una ' + label_with_distance)
                    engine.say('Precaucion, una ' +  label_with_distance)
                    engine.runAndWait()

                    detected_persons.append(person_id)
            
                

        label_with_distance = f'{label} {confidence} Está a una distancia de {round(distance, 2)} metros'
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label_with_distance, (x, y - 5), font, 1, color, 1)

        print(label_with_distance)

    cv2.imshow('YOLO Object Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()