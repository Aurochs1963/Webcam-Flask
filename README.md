# Projet Flask Webcam PC / PiCamera2
🔹 Dépendances principales

flask

opencv-python

picamera2 (si Raspberry Pi)

openai (si tu utilises l’analyse IA)

🔹 Structure projet

Webcam-Flask/\
│── app.py\
│── keys.py\
│── templates/\
│   ├── index.html        (page avec la webcam + boutons capture/analyse)\
│   └── analyze.html      (analyse + grille interactive + OpenAI)\
│── static/               (images capturées et extraites)\
│── models/               (modèles d'analyse des images (Yolo, mediapipe,...))\
│── uploads/              (vidéos uploadées)\

🔹 Routes Flask

/ → page d’accueil avec le flux vidéo (generate_frames()) + boutons

/video_feed → flux vidéo MJPEG (utilisé dans <img src> pour afficher la caméra en direct)

/capture → capture une image via get_frame() et la sauvegarde dans static/captures/

/gallery → affiche toutes les images du dossier static/captures/

/delete/<filename> → supprime une capture de la galerie

/analyze/<filename> → affiche la page d’analyse d’une capture avec la grille

/ask_openai/<filename> → envoie l’image + question texte à OpenAI (vision API) et retourne la réponse

/save_extraction → reçoit une zone sélectionnée (base64) et la sauvegarde dans static/captures/

🔹 Caméra (portable PC ↔ Pi)

Fonction get_frame() → retourne une image depuis :

Picamera2 (picam2.capture_array()) si dispo

sinon OpenCV (cv2.VideoCapture(0))

Utilisée dans :

generate_frames() (flux live)

/capture (sauvegarde d’image)

🔹 Grille interactive (analyse.html)

Grille paramétrable (ex. 4×4, 8×8) superposée sur l’image.

Numérotation automatique des cases.

Sélection rectangulaire cliquer-glisser avec la souris.

Extraction d’image côté client avec Canvas.

Sauvegarde côté serveur via /save_extraction.

Chaque extraction affichée avec bouton ⬇ Télécharger.

🔹 Intégration OpenAI

Upload image capturée + question utilisateur.

Envoi à gpt-4o-mini ou gpt-4o avec messages multimodaux :

client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Tu es un assistant d'analyse d'image"},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
        ]}
    ]
)

🔹 Checklist rapide (dev)


 Vérifier que picamera2 est installé sur Raspberry Pi

 Vérifier que app.secret_key est défini (si tu utilises des formulaires)

 Vérifier que la clef openAI (OPENAI_API_KEY) est bien renseignée dans keys.py si tu veux utiliser les API openAI pour l'analyse d'image


 Utiliser le fichier flask_app_anaconda.yaml pour créer un environnement virtuel conda avec toutes les dépendances


