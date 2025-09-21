# 📁 Projet Flask Webcam PC / PiCamera2 : Vision AI Toolkit - Documentation

## Description Générale

Cette application Flask est une plateforme web complète de **vision par ordinateur (computer vision)** et de **traitement d'images**. Elle permet de :

* Capturer des images et vidéos en direct depuis une webcam USB ou une caméra Raspberry Pi
* Appliquer en temps réel divers filtres et algorithmes de vision (détection de mouvement, suivi d'objets, YOLO, MediaPipe, etc.)
* Analyser des images statiques avec une multitude de traitements
* Interagir avec les images via l'IA générative (GPT-4o) de OpenAI
* Contrôler des servomoteurs pour orienter une caméra (sur Raspberry Pi)

## Structure des Fichiers

```
/project-root
│
├── app.py                 # Application Flask principale
├── keys.py               # Fichier de configuration (clés API, mots de passe)
├── models/               # Répertoire contenant les modèles de ML pré-entraînés
│   ├── deploy.prototxt.txt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── efficientdet_lite0.tflite
│   └── yolo11n.pt
├── static/               # Dossier pour les images uploadées/capturées
├── uploads/              # Dossier pour les vidéos uploadées
└── templates/            # Dossier des templates HTML
    ├── index.html
    └── analyze.html
```

## Dépendances et Installation

### Clés API (à définir dans `keys.py`)

```python
OPENAI_API_KEY = "votre_clé_openai"
OPENAI_ASSISTANT_ID = "votre_assistant_id"  # (Non utilisé dans ce code)
APP_PASSWORD_HASH = "hash_du_mot_de_passe"  # (Non utilisé dans ce code)
```

### Installation des Bibliothèques

```bash
pip install flask werkzeug opencv-python numpy openai picamera2 mediapipe ultralytics
```

## Fonctionnalités Principales et Routes Flask

### A. Flux Vidéo et Capture

| Route | Méthode | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Page d'accueil. Affiche la galerie des images capturées. |
| `/video` | `GET` | Flux vidéo MJPEG en direct. |
| `/capture` | `POST` | Capture une image depuis le flux vidéo et l'enregistre. |
| `/set_camera` | `GET` | Change la source vidéo (index de la caméra). |
| `/set_video` | `POST` | Upload et utilise un fichier vidéo comme source. |
| `/set_filter?mode=<mode>` | `GET` | Change le filtre appliqué au flux vidéo en direct. |

**Modes de Filtre Disponibles (`current_filter`):**

* `normal`: Flux vidéo brut
* `motion`: Détection de mouvement
* `track`: Suivi d'objet (CSRT tracker). Définir la ROI via `/set_roi_web`
* `yolo`: Détection d'objets avec YOLOv11
* `mediapipe`: Détection d'objets avec MediaPipe
* `pose`: Détection de points clés du corps humain
* `hands`: Détection des mains et reconnaissance de gestes

### B. Gestion des Images

| Route | Méthode | Description |
| :--- | :--- | :--- |
| `/download/<filename>` | `GET` | Télécharge une image. |
| `/delete/<filename>` | `POST` | Supprime une image. |
| `/upload` | `POST` | Upload une image depuis son ordinateur. |
| `/save_extraction` | `POST` | Sauvegarde une image extraite du canvas (en base64). |

### C. Analyse d'Image et IA

| Route | Méthode | Description |
| :--- | :--- | :--- |
| `/analyze/<filename>` | `GET`, `POST` | Applique un traitement à une image. |
| `/ask_openai/<filename>` | `POST` | Pose une question à GPT-4o à propos de l'image. |

**Méthodes d'Analyse (`method`):**

* `edges` (Contours)
* `gray` (Niveaux de gris)
* `blur` (Flou gaussien)
* `contrast` (Amélioration du contraste - CLAHE)
* `sharpen` (Accentuation)
* `faces` (Détection de visages - DNN)
* `mediapipe` (Détection d'objets - MediaPipe)
* `yolo` (Détection d'objets - YOLO)

### D. Contrôle Matériel (Raspberry Pi uniquement)

| Route | Méthode | Description |
| :--- | :--- | :--- |
| `/move_camera/<direction>` | `POST` | Contrôle les servomoteurs de la caméra. |
| `direction` = `up`, `down`, `left`, `right`, `reset` | | |

## Algorithmes de Vision Implémentés

### Détection de Mouvement
Utilise la soustraction d'images entre frames successifs et seuillage pour détecter les zones en mouvement.

### Suivi d'Objet (Tracking)
Initialise un tracker CSRT sur une région d'intérêt (ROI) définie par l'utilisateur et suit l'objet dans les frames suivants.

### Détection de Visages
Réseau de neurones profond (Caffe) pré-entraîné pour détecter les visages avec un score de confiance.

### Détection d'Objets (YOLO)
Utilise le modèle YOLOv11-nano (Ultralytics) pour détecter et localiser une large gamme d'objets (80 classes COCO).

### Détection d'Objets (MediaPipe)
Utilise le modèle EfficientDet-Lite0 de MediaPipe pour une détection d'objets rapide et précise.

### Détection de Pose
Utilise MediaPipe Pose pour estimer la pose humaine et dessiner les points clés et connections du corps.

### Détection de Mains et Gestes
Utilise MediaPipe Hands pour détecter les mains, les points de repère des doigts et reconnaître des gestes simples comme "Pouce levé", "Victoire", etc.

## Configuration et Exécution

### Sur un PC Standard (Webcam USB)
```bash
python app.py
```
* L'application utilise `cv2.VideoCapture(0)`
* Le mode debug est activé (`debug=True`)

### Sur un Raspberry Pi (Module caméra)
* Le script détecte automatiquement la disponibilité de `picamera2`
* Il initialise les servomoteurs sur les broches 8 et 11
* Le mode debug est désactivé pour éviter les conflits
* L'application se lance sur `0.0.0.0:5000`

## Architecture Technique

1. **Initialisation** : L'app vérifie le matériel et charge les modèles ML
2. **Flux Vidéo** : La fonction `generate_frames()` est un générateur qui produit un flux MJPEG en appliquant le filtre actif
3. **Traitements** : Les fonctions de traitement (e.g., `detect_objects_yolo()`) sont appelées en fonction du mode sélectionné et modifient le frame avant son envoi
4. **Interface Web** : Les routes Flask gèrent les interactions utilisateur (clics, uploads, formulaires) et mettent à jour l'état de l'application
5. **Session** : L'historique de chat avec OpenAI est stocké côté serveur dans l'objet `session` de Flask

## Notes et Points d'Attention

* **Performances** : Les modèles lourds (YOLO, MediaPipe) peuvent être gourmands en ressources. Les performances en temps réel dépendront de votre hardware
* **Raspberry Pi** : L'utilisation de `picamera2` est bien plus optimisée sur Pi qu'OpenCV avec une webcam USB
* **Sécurité** : L'application est conçue pour un usage en réseau local. Pour un déploiement public, renforcez la sécurité (HTTPS, authentification, validation des uploads)
* **Clés API** : La clé OpenAI est embarquée dans le code. Pour plus de sécurité, utilisez des variables d'environnement

Cette application est un excellent couteau suisse pour expérimenter avec la vision par ordinateur et l'IA !



