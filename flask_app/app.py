"""
Application Flask pour l'OCR de panneaux routiers
Interface web pour upload d'images et vidéos
"""
from flask import Flask, render_template, request, jsonify, send_file, url_for
import cv2
import numpy as np
import pytesseract
import os
import io
import base64
from PIL import Image
from werkzeug.utils import secure_filename
import tempfile
from pathlib import Path
import re
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Créer les dossiers s'ils n'existent pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Extensions autorisées
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, file_type='image'):
    """Vérifie si le fichier est autorisé"""
    if file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS
    return False

def detect_text_simple(image):
    """
    Détection OCR simple (sans YOLO pour l'instant)
    Retourne les zones de texte trouvées
    """
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Améliorer l'image pour l'OCR
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR avec données de position
    data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
    
    detections = []
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        confidence = int(data['conf'][i])
        text = data['text'][i].strip()
        
        if confidence > 60 and len(text) > 2:  # Filtrer les détections de qualité
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i] 
            h = data['height'][i]
            
            # Nettoyer le texte
            clean_text = re.sub(r'[^A-Za-z0-9\s-]', '', text)
            
            if len(clean_text.strip()) > 1:
                detections.append({
                    'text': clean_text.strip(),
                    'bbox': [x, y, w, h],
                    'confidence': confidence
                })
    
    return detections

def annotate_image(image, detections):
    """Annote l'image avec les détections"""
    annotated = image.copy()
    
    for detection in detections:
        text = detection['text']
        x, y, w, h = detection['bbox']
        confidence = detection['confidence']
        
        # Dessiner le rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Préparer le texte à afficher
        label = f"{text} ({confidence}%)"
        
        # Calculer la taille du texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Dessiner le fond du texte
        cv2.rectangle(annotated, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
        
        # Dessiner le texte
        cv2.putText(annotated, label, (x, y - 5), font, font_scale, (0, 0, 0), thickness)
    
    return annotated

@app.route('/')
def index():
    """Page d'accueil avec interface upload"""
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Traite l'upload d'image"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if not allowed_file(file.filename, 'image'):
        return jsonify({'error': 'Format de fichier non supporté'}), 400
    
    try:
        # Lire l'image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Impossible de lire l\'image'}), 400
        
        # Détecter le texte
        detections = detect_text_simple(image)
        
        # Annoter l'image
        annotated_image = annotate_image(image, detections)
        
        # Encoder l'image annotée en base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Préparer la réponse
        response = {
            'success': True,
            'detections': detections,
            'annotated_image': f"data:image/jpeg;base64,{img_base64}",
            'total_detections': len(detections)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement: {str(e)}'}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Traite l'upload de vidéo"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if not allowed_file(file.filename, 'video'):
        return jsonify({'error': 'Format de vidéo non supporté'}), 400
    
    try:
        # Sauvegarder le fichier temporairement
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Traiter la vidéo
        output_filename = f"annotated_{unique_filename}"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        total_detections = process_video(input_path, output_path)
        
        # Nettoyer le fichier d'entrée
        os.remove(input_path)
        
        response = {
            'success': True,
            'output_filename': output_filename,
            'total_detections': total_detections,
            'download_url': url_for('download_video', filename=output_filename)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement: {str(e)}'}), 500

def process_video(input_path, output_path, max_frames=300):
    """
    Traite une vidéo frame par frame
    Limite à max_frames pour éviter les timeouts
    """
    cap = cv2.VideoCapture(input_path)
    
    # Propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limiter le nombre de frames
    frames_to_process = min(total_frames, max_frames)
    frame_step = max(1, total_frames // frames_to_process)
    
    # Writer pour la vidéo de sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_detections = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traiter seulement certaines frames
            if frame_count % frame_step == 0:
                detections = detect_text_simple(frame)
                annotated_frame = annotate_image(frame, detections)
                total_detections += len(detections)
                out.write(annotated_frame)
            else:
                out.write(frame)
            
            frame_count += 1
            
            # Limiter le nombre de frames
            if frame_count >= frames_to_process:
                break
                
    finally:
        cap.release()
        out.release()
    
    return total_detections

@app.route('/download/<filename>')
def download_video(filename):
    """Télécharge la vidéo traitée"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Fichier non trouvé", 404

@app.route('/health')
def health():
    """Endpoint de santé"""
    return jsonify({'status': 'OK', 'service': 'OCR Road Signs API'})

if __name__ == '__main__':
    print("🚀 Démarrage de l'API Flask OCR")
    print("Interface disponible sur: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)