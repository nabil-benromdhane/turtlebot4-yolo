from ultralytics import YOLO
import cv2
import sys

# Charger le modèle entraîné
model = YOLO("results/best.pt")

# Classes du dataset
classes = ['TV', 'bed', 'chair', 'clock', 'counter', 
           'door', 'fan', 'light', 'sofa', 'switchboard', 'table']

def test_image(image_path):
    """Tester sur une image"""
    print(f"Test sur : {image_path}")
    
    # Détection
    results = model(image_path)
    
    # Afficher les résultats
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("Aucun objet détecté")
        else:
            print(f"{len(boxes)} objet(s) détecté(s) :")
            for box in boxes:
                classe = classes[int(box.cls)]
                confiance = float(box.conf)
                print(f"  → {classe} ({confiance:.0%})")
    
    # Sauvegarder l'image avec les détections
    results[0].save("results/detection_result.jpg")
    print("Image sauvegardée dans results/detection_result.jpg ✅")

def test_webcam():
    """Tester en temps réel avec webcam"""
    print("Test webcam — appuie sur 'q' pour quitter")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Détection sur chaque frame
        results = model(frame, verbose=False)
        
        # Afficher le résultat
        annotated = results[0].plot()
        cv2.imshow("YOLO Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Par défaut : tester sur une image
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "webcam":
            test_webcam()
        else:
            test_image(sys.argv[1])
    else:
        print("Usage:")
        print("  python3 test.py image.jpg     ← tester sur une image")
        print("  python3 test.py webcam        ← tester en temps réel")
