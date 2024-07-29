import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Cargar el modelo
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # 1 clase (persona) + fondo
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Cargar los pesos guardados
model.load_state_dict(torch.load('fasterrcnn_person_detector.pth'))
model.eval()

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def predict_and_draw_boxes_video(video_path, model, device, threshold=0.5):
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir el frame a formato PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Transformar la imagen
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
        
        # Realizar la predicción
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Filtrar las detecciones por umbral de confianza
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        
        # Contar las personas detectadas
        num_people = sum(
            1 for i, label in enumerate(pred_labels) 
            if label == 1 and pred_scores[i] >= threshold
        )

        # Dibujar los cuadros delimitadores y el texto
        for i, box in enumerate(pred_boxes):
            if pred_scores[i] >= threshold:
                (x1, y1, x2, y2) = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Añadir el texto con la cantidad de personas detectadas
        text = f'People Detected: {num_people}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Mostrar el frame con los cuadros delimitadores y el texto
        cv2.imshow('Person Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Realizar la predicción y dibujar los cuadros delimitadores en un video
    video_path = 0  # usa 0 para la cámara de la laptop o la ruta de un archivo de video
    # video_path  = '/home/russell/git/CountPeople/video/videoTest.mp4'

    predict_and_draw_boxes_video(video_path, model, device)
