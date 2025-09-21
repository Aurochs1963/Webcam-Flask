from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory,session,jsonify
from werkzeug.utils import secure_filename

import cv2, urllib.request
import numpy as np

import os
import time

from openai import OpenAI
import base64

from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID,APP_PASSWORD_HASH

import importlib

app = Flask(__name__)

app.secret_key = "my_prefered_secret_key"  # clé pour sécuriser la session

client = OpenAI(api_key=OPENAI_API_KEY)

# Vérifier si picamera2 est dispo
try:
    picam2_module = importlib.util.find_spec("picamera2")
    if picam2_module is not None:
        from picamera2 import Picamera2,Preview
        from robot_hat import Servo
        USE_PICAMERA2 = True
    else:
        USE_PICAMERA2 = False
except ImportError:
    USE_PICAMERA2 = False

angleHB = 0
angleDG = 0

# Initialisation caméra
if USE_PICAMERA2:
    print("Utilisation de picamera2 (Raspberry Pi Camera)")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))
    picam2.start()

    def get_frame():
        frame = picam2.capture_array()
        return frame

    def motor_up():
        global angleHB
        angleHB = angleHB - 5
        #print("angleHB",angleHB)
        Servo(8).angle(angleHB)

    def motor_down():
        global angleHB
        angleHB = angleHB + 5
        #print("angleHB",angleHB)
        Servo(8).angle(angleHB)

    def motor_left():
        global angleDG
        angleDG = angleDG - 5
        #print("angleDG",angleDG)
        Servo(11).angle(angleDG)

    def motor_right():
        global angleDG
        angleDG = angleDG + 5
        #print("angleDG",angleDG)
        Servo(11).angle(angleDG)
  
    def reset_cam():
        Servo(8).angle(0)
        Servo(11).angle(0)
else:
    print("Utilisation de cv2.VideoCapture (webcam USB)")
    camera = cv2.VideoCapture(0)  # 0 = caméra avant , 1 = Caméra arriere

    def get_frame():
        global video_file_mode
        ret, frame = camera.read()
        if not ret:
            # Si fichier vidéo terminé → recommencer
            if video_file_mode:  # variable globale à définir
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                return None

        return frame

video_file_mode = 1
current_filter = "normal"
prev_frame = None
tracker = None
roi_box = None

@app.route("/set_camera")
def set_camera():
    global camera
    index = int(request.args.get("index", 0))
    
    # Fermer la caméra en cours
    if camera is not None:
        camera.release()
    
    # Ouvrir la nouvelle caméra
    camera = cv2.VideoCapture(index)
    return ("", 204)

@app.route("/set_video", methods=["POST"])
def set_video():
    global camera
    video_file = request.files["video"]
    path = os.path.join("uploads", video_file.filename)
    os.makedirs("uploads", exist_ok=True)
    video_file.save(path)

    if camera is not None:
        camera.release()
    camera = cv2.VideoCapture(path)
    return ("", 204)
    
@app.route("/set_filter")
def set_filter():
    global current_filter
    mode = request.args.get("mode", "normal")
    current_filter = mode
    return ("", 204)

@app.route("/set_roi_web", methods=["POST"])
def set_roi_web():
    global roi_box,tracker
    data = request.get_json()
    # Coordonnées normalisées (par rapport à la taille de l'image affichée)
    roi_box = (data["x"], data["y"], data["w"], data["h"],
               data["img_w"], data["img_h"])
    tracker = None
    return jsonify({"status": "ok"})

# Génération du flux pour Flask
def generate_frames():
    global prev_frame, tracker, current_filter,roi_box
    
    while True:
        frame = get_frame()
        if frame is None:
            continue

        if current_filter == "motion":
            # Détection de mouvement (comme déjà fait)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
            else:
                diff = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 500:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                prev_frame = gray

        elif current_filter == "track":
            if roi_box is not None and tracker is None:
                # Adapter ROI aux dimensions réelles du flux vidéo
                frame_h, frame_w = frame.shape[:2]
                x, y, w, h, img_w, img_h = roi_box
                scale_x = frame_w / img_w
                scale_y = frame_h / img_h
                roi_scaled = (int(x*scale_x), int(y*scale_y),
                              int(w*scale_x), int(h*scale_y))

                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, roi_scaled)
                roi_box = None  # reset après init

            elif tracker is not None:
                success, box = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                else:
                    tracker = None
 
        elif current_filter == "yolo":
            frame,detections = detect_objects_yolo(frame)

        elif current_filter == "mediapipe":
            frame,detections = detect_objects_mediapipe(frame)
                     
        elif current_filter == "pose":
            frame = apply_pose_detection(frame)
            
        elif current_filter == "hands":
            frame = apply_hand_detection(frame)
    
        # Encodage en JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    captures = sorted(
        [f for f in os.listdir("static") if not f.startswith("analyzed_")],
        reverse=True
    )
    return render_template('index.html', captures=captures)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    frame = get_frame()
    if frame is None:
        return "Erreur capture", 500
    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join("static", filename)
    cv2.imwrite(filepath, frame)
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory("static", filename, as_attachment=True)
    
@app.route('/delete/<filename>', methods=['POST'])
def delete(filename):
    filepath = os.path.join("static", filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join("static", filename)
        file.save(filepath)

    return redirect(url_for('index'))

CAPTURES_DIR = "static"

@app.route("/save_extraction", methods=["POST"])
def save_extraction():
    data = request.json.get("image")
    if not data.startswith("data:image/png;base64,"):
        return jsonify({"error": "Format invalide"}), 400

    # Nettoyer l'encodage base64
    image_data = data.replace("data:image/png;base64,", "")
    image_bytes = base64.b64decode(image_data)

    # Nom du fichier unique
    filename = f"extraction_{int(time.time())}.png"
    filepath = os.path.join(CAPTURES_DIR, filename)

    # Sauvegarde
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return jsonify({"path": f"/{filepath}"})

@app.route("/move_camera/<direction>", methods=["POST"])
def move_camera(direction):
    try:
        if direction == "up":
            motor_up()
        elif direction == "down":
            motor_down()
        elif direction == "left":
            motor_left()
        elif direction == "right":
            motor_right()
        elif direction == "reset":
            reset_cam()
        else:
            return jsonify({"status":"error","message":"direction inconnue"}), 400
        return jsonify({"status":"ok","direction":direction})
    except Exception as e:
        print(str(e))
        return jsonify({"status":"error","message":str(e)}), 500

# Détection des visages
# Charger le réseau DNN
prototxt = "models/deploy.prototxt.txt"
model = "models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

def detect_faces_dnn(frame, conf_threshold=0.6):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces

# Détection des objets
# mediapipe (google)

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_object_detector():
    base_options = python.BaseOptions(model_asset_path="models/efficientdet_lite0.tflite")
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.3)
    detector = vision.ObjectDetector.create_from_options(options)
    return detector

detector = load_object_detector()

def detect_objects_mediapipe(frame):
    # Convertir OpenCV (BGR) → Mediapipe (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    detections = []
    h, w, _ = frame.shape
    for det in result.detections:
        category = det.categories[0]
        bbox = det.bounding_box

        x_min = int(bbox.origin_x)
        y_min = int(bbox.origin_y)
        x_max = int(bbox.origin_x + bbox.width)
        y_max = int(bbox.origin_y + bbox.height)

        label = f"{category.category_name} ({category.score:.2f})"
        detections.append((label, (x_min, y_min, x_max, y_max)))

        # Dessiner sur l’image
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
        # Using cv2.putText() method
        #cv2.putText(image, 'OpenCV', org, font,fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, label, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 
    return frame, detections

#Detection des poses

import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def apply_pose_detection(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
        )
    return frame
    
#Détection des mains

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def fingers_up(hand_landmarks):
    fingers = []

    # Index, majeur, annulaire, auriculaire
    tips = [8, 12, 16, 20]  # bouts des doigts
    bases = [6, 10, 14, 18] # articulations
    
    for tip, base in zip(tips, bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            fingers.append(1)  # doigt levé
        else:
            fingers.append(0)

    # Pouce (différent car horizontal)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.insert(0, 1)
    else:
        fingers.insert(0, 0)

    return fingers  # ex: [1,0,0,0,0] = pouce levé


def detect_gesture(hand_landmarks):
    fingers = fingers_up(hand_landmarks)

    if fingers == [1,0,0,0,0]:
        return "Pouce leve"
    elif fingers == [0,1,1,0,0]:
        return "Victoire"
    elif fingers == [0,0,1,0,0]:
        return "DTC"
    elif fingers == [1,1,1,1,1]:
        return "Main ouverte"
    elif fingers == [0,1,0,0,1]:
        return "Rock'n roll !"
    else:
        return None


def apply_hand_detection(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            gesture = detect_gesture(hand_landmarks)
            if gesture:
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(frame, gesture, (x, y-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame
    
#yolo

from ultralytics import YOLO

# Charger un modèle YOLO pré-entraîné (80 classes COCO)
# 'yolov8n.pt' = nano (rapide, léger)
# 'yolov8s.pt' = small (plus précis, mais plus lent)
yolo_model = YOLO("models/yolo11n.pt")

detections = []  # stocker les boîtes détectées
trackers = []    # liste de trackers actifs

def detect_objects_yolo(frame):
    # YOLO attend du RGB
    results = yolo_model.predict(frame, imgsz=640, conf=0.4, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            confidence = float(box.conf[0])
            (x1, y1, x2, y2) = box.xyxy[0].int().tolist()

            detections.append((f"{label} ({confidence:.2f})", (x1, y1, x2, y2)))

            # Dessiner sur l'image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255), 2)

    return frame, detections

# traitement des images opencv
def enhance_contrast(frame):
    # Conversion en LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE sur la composante L (luminosité)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced, [("contrast_enhanced", (0,0,0,0))]  # liste vide pour cohérence

def sharpen_image(frame):
    # Noyau de netteté
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened, [("sharpened", (0,0,0,0))]
    
@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze(filename):
    filepath = os.path.join("static", filename)
    analyzed_filename = f"analyzed_{filename}"
    analyzed_path = os.path.join("static", analyzed_filename)

    method = request.form.get("method", "gray")  # Par défaut : contours

    detections = []

    # Lire l'image
    image = cv2.imread(filepath)
    if image is None:
        return f"Impossible de lire {filename}", 404

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = image.copy()

    if method == "edges":
        edges = cv2.Canny(gray, 100, 200)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif method == "gray":
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif method == "blur":
        result = cv2.GaussianBlur(image, (15, 15), 0)

    elif method == "contrast":
        result, detections = enhance_contrast(result)

    elif method == "sharpen":
        result, detections = sharpen_image(result)

    elif method == "faces":
        faces = detect_faces_dnn(result, conf_threshold=0.6)
        for (x, y, x2, y2) in faces:
            cv2.rectangle(result, (x, y), (x2, y2), (0, 255, 0), 2)
            detections.append(("face", (x, y, x2, y2)))
            
    elif method == "mediapipe":
        result, detections = detect_objects_mediapipe(result)
    
    elif method == "yolo":
        result, detections = detect_objects_yolo(result)
 
    # Sauvegarder
    cv2.imwrite(analyzed_path, result)

    return render_template('analyze.html', filename=filename, analyzed=analyzed_filename, method=method, detections=detections)
    
@app.route('/ask_openai/<filename>', methods=['POST'])
def ask_openai(filename):
    question = request.form.get("question")
    filepath = os.path.join("static", filename)

    # Charger l'image en base64
    with open(filepath, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")


    # Initialiser l'historique pour cette image
    if "chat_history" not in session:
        session["chat_history"] = {}

    if filename not in session["chat_history"]:
        session["chat_history"][filename] = []

    # Construire la conversation passée
    messages = [{"role": "system", "content": "Tu es un assistant qui décrit et analyse des images."}]
    for entry in session["chat_history"][filename]:
        messages.append({"role": "user", "content": [{"type": "text", "text": entry["question"]}]})
        messages.append({"role": "assistant", "content": entry["answer"]})

    # Ajouter la nouvelle question
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    })

    # Appel OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Sauvegarder dans l'historique
    session["chat_history"][filename].append({
        "question": question,
        "answer": answer
    })
    session.modified = True

    return render_template(
        'analyze.html',
        filename=filename,
        analyzed=f"analyzed_{filename}",
        method="",
        chat_history=session["chat_history"][filename]
    )
    
if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    # Remplace '0.0.0.0' par l'IP locale de ta machine si tu veux être plus restrictif
    if USE_PICAMERA2:
       app.run(host="0.0.0.0", port=5000, debug=False)
    else:
       app.run(host="0.0.0.0", port=5000, debug=True) # Debug True ne marche pas avec la picamera , conflit
