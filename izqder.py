import cv2
import numpy as np
import pyttsx3

weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'

detected_persons = []
net = cv2.dnn.readNet(weights_path, config_path)

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in np.array(indexes).flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        focal_length = 300  
        real_height = 1.0 
        pixel_height = h  
        distance = (focal_length * real_height) / pixel_height
        label_with_distance = f'{label} ' #esta a una distancia de {round(distance, 2)} metros'

        # Calcula el centro horizontal del objeto y de la imagen
        center_image_x = width / 2
        center_object_x = x + w / 2

        # Compara para determinar la posición del objeto
        if center_object_x < center_image_x:
            position = 'a la izquierda'
        elif center_object_x > center_image_x:
            position = 'a la derecha'
        else:
            position = 'en el centro'

        person_id = f'{class_ids[i]}_{boxes[i]}'
        if person_id not in detected_persons:
            if label == 'botella':
                speech = f'Esto es una {label_with_distance}'
            elif label == 'senal de alto':
                speech = f'Precaucion, hay una {label_with_distance}'
            elif label == 'semaforo':
                speech = f'Precaucion, te acercas a un {label_with_distance}'
            elif label == 'silla' and distance < 2.0:
                speech = f'Precaucion, una {label_with_distance}'
            elif label == 'tenedor':
                speech = f'Esto es un {label_with_distance}'
            else:
                speech = ''   # Definir una cadena vacía si no se satisface ninguna condición
            
            if speech:
                engine = pyttsx3.init()
                print(speech)
                engine.say(speech)
                engine.runAndWait()
                detected_persons.append(person_id)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label_with_distance + f' ({position})', (x, y - 5), font, 1, color, 1)

    cv2.imshow('V-My Eyes', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
