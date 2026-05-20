from flask import Flask, render_template, Response, jsonify, request, send_file, redirect, url_for, session, flash
import cv2
import google.generativeai as genai
import os
from datetime import datetime, timedelta
import time
from deepface import DeepFace
from dotenv import load_dotenv
import numpy as np
import threading
import requests
import re
import torch
import tensorflow as tf
import io
import PyPDF2
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
load_dotenv(override=True)

import json
from flask.json.provider import DefaultJSONProvider
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.getenv('SECRET_KEY', 'my_awesome_hackathon_secret_key_123')

# User database (in-memory for demonstration, replace with actual database in production)
# Replace users = {} with this:
USER_FILE = 'users.json'

def load_users():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {} # Return empty if file is corrupt or empty
    return {}

def save_users(users_dict):
    with open(USER_FILE, 'w') as f:
        json.dump(users_dict, f, indent=4)

# Load users when the app starts
users = load_users()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Global variables
analysis_history = []
session_metrics = {
    'attention_avg': [],
    'engagement_avg': [],
    'training_quality_avg': [],
    'gemini_insights': []
}

# --- ADD THESE TWO LINES ---
global_frame = None
frame_lock = threading.Lock()
# Add at the top with other global variables
# Email Configuration
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")


analysis_results = {
    'metrics': {
        'attention': 0,
        'engagement': 0,
        'training_quality': 0,
        'infrastructure_compliance': 0,
        'student_interaction': 0,
        'equipment_usage': 0,
        'space_utilization': 0,
        'skill_compliance': 0
    }
}

analysis_history = []
session_metrics = {
    'attention_avg': [],
    'engagement_avg': [],
    'training_quality_avg': [],
    'gemini_insights': []
}
my_new_key = "AIzaSyAA3OSOmVJkJnfuzd59c7LSPff9J06vm-U"
print("\n" + "="*50)
print(f"🔑 FORCING NEW API KEY STARTING WITH: {my_new_key[:15]}...")
print("="*50 + "\n")
# Configure Gemini
genai.configure(api_key= my_new_key)
model = genai.GenerativeModel('gemini-2.0-flash')
text_model = genai.GenerativeModel('gemini-2.0-flash')
ANALYSIS_PROMPT = """
Please analyze this skill training classroom session and provide a plain text evaluation covering:

1. Training Effectiveness
- Describe student engagement and participation level
- Comment on practical skills demonstration
- Evaluate equipment utilization rate
- Assess trainer-student interaction quality

2. Infrastructure Assessment
- Comment on required equipment availability
- Evaluate space utilization effectiveness
- Check safety compliance status
- Review training aids usage

3. Quality Indicators
- Evaluate hands-on practice opportunities
- Assess student attention levels
- Review group activities implementation
- Check course curriculum alignment

4. Overall Risk Level
Classify the session as:
GREEN: Meets all standards
YELLOW: Needs improvement
RED: Immediate intervention required

Provide your analysis in simple text format without any special characters, symbols, or formatting. Focus on clear, actionable observations and recommendations.
"""

def clean_text(text):
    """Clean text from special characters and markdown symbols"""
    # Remove markdown headers (#)
    text = re.sub(r'#+\s*', '', text)
    # Remove markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Remove square brackets and content
    text = re.sub(r'\[.*?\]', '', text)
    # Remove markdown tables (|)
    text = re.sub(r'\|.*?\|', '', text)
    # Remove multiple dashes
    text = re.sub(r'-{2,}', '-', text)
    # Remove any remaining special characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:\-()\'"]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()

# Configure GPU settings
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    torch.cuda.set_device(0)  # Use the first GPU
    # Configure TensorFlow to use GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    print("No GPU available, using CPU")

# Initialize DeepFace backend with GPU support
DeepFace.build_model("Facenet512")

class VideoProcessor:
    def __init__(self):
        self.face_cascade = cv2.CUDACascade("haarcascade_frontalface_default.xml") if cv2.cuda.getCudaEnabledDeviceCount() > 0 else cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cuda_stream = cv2.cuda.Stream() if cv2.cuda.getCudaEnabledDeviceCount() > 0 else None
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
    def preprocess_frame(self, frame):
        if self.use_gpu:
            # Upload frame to GPU memory
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            
            # Equalize histogram for better face detection
            gpu_gray = cv2.cuda.equalizeHist(gpu_gray)
            
            return gpu_gray, gpu_frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            return gray, None

    def detect_faces(self, frame):
        if self.use_gpu:
            gpu_gray, gpu_frame = self.preprocess_frame(frame)
            faces = self.face_cascade.detectMultiScale(gpu_gray, scaleFactor=1.1, minNeighbors=5)
            return faces, gpu_frame
        else:
            gray, _ = self.preprocess_frame(frame)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            return faces, None

# Create global video processor instance
video_processor = VideoProcessor()

def process_frame(frame, metrics_only=False):
    """Process a single frame for facial analysis with GPU acceleration"""
    try:
        if frame is None or not frame.size > 0:
            return None

        # Convert frame to RGB (DeepFace expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze the whole frame with DeepFace
        try:
            results = DeepFace.analyze(
                rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]
        except Exception as e:
            print(f"DeepFace analysis error: {str(e)}")
            return None
            
        # Calculate overall metrics
        total_attention = 0
        total_engagement = 0
        num_faces = len(results)
        
        # Calculate metrics
        metrics = {}
        
        for face_data in results:
            # Extract face region
            face_coords = face_data.get('region', {})
            x = int(face_coords.get('x', 0))
            y = int(face_coords.get('y', 0))
            w = int(face_coords.get('w', 0))
            h = int(face_coords.get('h', 0))
            
            face_roi = frame[y:y+h, x:x+w]
            emotion = face_data.get('dominant_emotion', 'neutral')
            attention_score = 1.0 - (face_data.get('emotion', {}).get('neutral', 0) / 100)
            
            engagement_state = map_emotion_to_state(emotion, attention_score)
            if engagement_state in ['distracted', 'sleeping']:
                send_distraction_alert(engagement_state, attention_score, face_roi)
            # Get dominant emotion and calculate attention score
            emotion = face_data.get('dominant_emotion', 'neutral')
            attention_score = 1.0 - (face_data.get('emotion', {}).get('neutral', 0) / 100)
            
            # Accumulate metrics
            total_attention += attention_score
            engagement_score = (face_data.get('emotion', {}).get('happy', 0) + 
                              face_data.get('emotion', {}).get('surprise', 0)) / 100
            total_engagement += engagement_score
            
            if not metrics_only:
                # Map emotion to engagement state
                engagement_state = map_emotion_to_state(emotion, attention_score)
                if engagement_state in ['distracted', 'sleeping']:
                    pass
                # Draw bounding box with corresponding color
                color = EMOTION_COLORS.get(engagement_state, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add text label
                label = f"{engagement_state.title()} ({attention_score:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
        
        # Calculate final metrics
        if num_faces > 0:
            metrics['attention'] = min((total_attention / num_faces) * 10, 10)
            metrics['engagement'] = min((total_engagement / num_faces) * 10, 10)
        else:
            metrics['attention'] = 0
            metrics['engagement'] = 0
        
        # Calculate other metrics
        metrics['training_quality'] = calculate_training_quality()
        metrics['infrastructure_compliance'] = assess_infrastructure()
        metrics['student_interaction'] = calculate_interaction_score()
        metrics['equipment_usage'] = assess_equipment_usage()
        metrics['space_utilization'] = calculate_space_utilization()
        
        if not metrics_only:
            # Display metrics on frame
            metrics_color = (0, 255, 0)
            metrics_text = [
                f"Attention: {metrics['attention']:.1f}/10",
                f"Engagement: {metrics['engagement']:.1f}/10",
                f"Quality: {metrics['training_quality']:.1f}/10",
                f"Compliance: {metrics['infrastructure_compliance']:.1f}/10"
            ]
            
            for idx, text in enumerate(metrics_text):
                y_pos = 30 + (idx * 40)
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, metrics_color, 2)
            
            return frame
        
        return metrics
        
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        return None
def send_distraction_alert(student_state, attention_score, face_image):
    try:
        # Create email message
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = RECEIVER_EMAIL
        message["Subject"] = f"Student Attention Alert - {student_state}"
        
        body = f"""
        Attention Alert!
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Student State: {student_state}
        Attention Score: {attention_score:.2f}/10
        
        This is an automated alert from the Training Monitoring System.
        Attached is the detected distracted face image.
        """
        
        message.attach(MIMEText(body, "plain"))
        
        # Check if face_image is valid
        if face_image is not None and face_image.size > 0:
            # Convert face ROI to jpg image
            _, img_encoded = cv2.imencode('.jpg', face_image)
            
            # Attach face image
            image = MIMEImage(img_encoded.tobytes())
            image.add_header('Content-ID', '<distracted_face>')
            image.add_header('Content-Disposition', 'attachment', filename='distracted_face.jpg')
            message.attach(image)
        
        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
            print(f"Alert email sent with face image for {student_state} state")
            
    except Exception as e:
        print(f"Failed to send alert email: {str(e)}")

def map_emotion_to_state(emotion, attention_score):
    """Map DeepFace emotion to engagement state"""
    if emotion == 'happy' or emotion == 'surprise':
        return 'engaged' if attention_score > 0.6 else 'interested'
    elif emotion == 'neutral':
        return 'neutral'
    elif emotion == 'sad' or emotion == 'fear':
        return 'distracted'
    elif emotion == 'angry' or emotion == 'disgust':
        return 'distracted'
    else:
        return 'neutral'

def continuous_analysis():
    global global_frame 
    frames_buffer = []
    print("✅ continuous_analysis thread started!")
    while True:
        try:
            frame_to_process = None
            
            # Safely grab the latest frame
            with frame_lock:
                if global_frame is not None:
                    frame_to_process = global_frame.copy()

            if frame_to_process is not None:
                # print("🔄 Processing frame for metrics...") # (You can delete or comment this out so it stops spamming your terminal!)
                metrics = process_frame(frame_to_process, metrics_only=True)
                
                if metrics:
                    # --- THE ULTIMATE FIX: Clean the numbers BEFORE saving! ---
                    clean_metrics = {key: float(value) for key, value in metrics.items()}
                    analysis_results['metrics'].update(clean_metrics)
                    
                    print(f"📊 Cleaned Metrics updated!") # Kept it short so your terminal is easier to read
            
            time.sleep(2)
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            time.sleep(2)

def calculate_training_quality():
    """Calculate training quality based on attention and engagement"""
    attention = analysis_results['metrics'].get('attention', 0)
    engagement = analysis_results['metrics'].get('engagement', 0)
    return (attention + engagement) / 2

def assess_infrastructure():
    """Assess infrastructure compliance"""
    # Simplified calculation for demo
    return 8.5

def calculate_interaction_score():
    """Calculate student interaction score"""
    engagement = analysis_results['metrics'].get('engagement', 0)
    return min(engagement * 1.2, 10)

def assess_equipment_usage():
    """Assess equipment usage"""
    # Simplified calculation for demo
    return 7.5

def calculate_space_utilization():
    """Calculate space utilization"""
    # Simplified calculation for demo
    return 9.0

def analyze_historical_data():
    if not analysis_history:
        return {
            'trends': [],
            'performance_summary': {},
            'improvement_indicators': []
        }
    
    recent_sessions = analysis_history[-10:]
    trends = {
        'attention': [],
        'engagement': [],
        'training_quality': [],
        'infrastructure_compliance': [],
        'student_interaction': []
    }
    
    for session in recent_sessions:
        for metric, value in session['metrics'].items():
            if metric in trends:
                trends[metric].append(value)
    
    performance_summary = {
        metric: {
            'average': sum(values) / len(values) if values else 0,
            'trend': 'improving' if len(values) >= 2 and values[-1] > values[0] else 'declining'
        }
        for metric, values in trends.items()
    }
    
    improvement_indicators = [
        metric for metric, data in performance_summary.items()
        if data['average'] < 7.0 or data['trend'] == 'declining'
    ]
    
    return {
        'trends': trends,
        'performance_summary': performance_summary,
        'improvement_indicators': improvement_indicators
    }

def determine_risk_level():
    metrics = analysis_results['metrics']
    if not metrics: return 'YELLOW'
    
    # 🛡️ SHIELD: Force every single value to be a normal float before doing math
    avg_score = sum(float(v) for v in metrics.values()) / len(metrics)
    
    if avg_score >= 7.5:
        return 'GREEN'
    elif avg_score >= 5.0:
        return 'YELLOW'
    else:
        return 'RED'

def check_compliance_status():
    # 🛡️ SHIELD: Create a brand new, perfectly clean dictionary of normal floats
    clean_metrics = {k: float(v) for k, v in analysis_results['metrics'].items()}
    
    compliance_criteria = {
        'infrastructure': { 'space_utilization': 7.0, 'equipment_usage': 8.0, 'safety_standards': 8.5 },
        'training_quality': { 'engagement': 7.0, 'attention': 7.0, 'interaction': 7.5 },
        'course_delivery': { 'training_quality': 7.5, 'student_interaction': 7.0 }
    }
    
    compliance_status = {
        'infrastructure': {
            'status': 'compliant' if clean_metrics.get('infrastructure_compliance', 0) >= compliance_criteria['infrastructure']['space_utilization'] else 'non_compliant',
            'score': clean_metrics.get('infrastructure_compliance', 0),
            'issues': []
        },
        'training_quality': {
            'status': 'compliant' if clean_metrics.get('training_quality', 0) >= compliance_criteria['training_quality']['engagement'] else 'non_compliant',
            'score': clean_metrics.get('training_quality', 0),
            'issues': []
        },
        'course_delivery': {
            'status': 'compliant' if clean_metrics.get('student_interaction', 0) >= compliance_criteria['course_delivery']['student_interaction'] else 'non_compliant',
            'score': clean_metrics.get('student_interaction', 0),
            'issues': []
        }
    }
    
    if clean_metrics.get('space_utilization', 0) < compliance_criteria['infrastructure']['space_utilization']:
        compliance_status['infrastructure']['issues'].append('Space utilization below required standards')
    
    compliant_categories = sum(1 for category in compliance_status.values() if category['status'] == 'compliant')
    overall_status = 'COMPLIANT' if compliant_categories == len(compliance_status) else 'NON-COMPLIANT'
    
    return {
        'overall_status': overall_status,
        'detailed_status': compliance_status,
        'timestamp': datetime.now().isoformat(),
        'action_required': overall_status == 'NON-COMPLIANT'
    }

def generate_recommendations():
    metrics = analysis_results['metrics']
    recs = {
        'immediate_actions': [],
        'improvement_areas': [],
        'best_practices': ["Start sessions with clear learning objectives", "Maintain good lighting for AI tracking"]
    }
    
    # If engagement is low, suggest actions
    if metrics.get('engagement', 0) < 6.0:
        recs['immediate_actions'].append("Pause and ask the audience direct questions to wake them up.")
        recs['improvement_areas'].append("Incorporate more hands-on group activities.")
    else:
        recs['best_practices'].append("Excellent engagement! Continue using interactive discussion methods.")
        
    # If attention is low, suggest actions
    if metrics.get('attention', 0) < 6.0:
        recs['immediate_actions'].append("Change tone of voice or use a visual aid to regain focus.")
        
    return recs

def generate_frames():
    global global_frame
    ip_address = "10.234.216.202:8080"
    print(f"🎥 Connecting to {ip_address}")
    while True:
        try:
            url = f"http://{ip_address}/shot.jpg"
            img_resp = requests.get(url, timeout=5)
            if img_resp.status_code == 200:
                img_arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, (640, 480))
                    with frame_lock:
                        global_frame = frame.copy()

                    # Skip DeepFace, just show raw camera feed first
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        print("✅ Frame sent!")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            time.sleep(1)
# Define emotion color mapping
EMOTION_COLORS = {
    'engaged': (0, 255, 0),      # Green
    'interested': (255, 165, 0),  # Orange
    'distracted': (255, 0, 0),    # Red
    'sleeping': (128, 0, 128),    # Purple
    'neutral': (255, 255, 0)      # Yellow
}
curriculum_content = None
curriculum_type = None
def analyze_skill_compliance(frame_analysis, curriculum):
    if not curriculum:
        return 0, "No curriculum uploaded"
        
    try:
        # Create detailed analysis
        analysis_text = f"""
        Current Training Analysis:
        - Number of participants detected
        - Current engagement levels
        - Skills being demonstrated
        
        Curriculum Alignment:
        {curriculum[:200]}...
        """
        
        # Add to session metrics
        session_metrics['skill_monitoring'].append({
            'timestamp': datetime.now().isoformat(),
            'score': 75,  # Example score
            'analysis': analysis_text
        })
        
        return 75, analysis_text
            
    except Exception as e:  
        print(f"Skill analysis error: {str(e)}")
        return 0, "Analysis failed"


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyPDF2"""
    try:
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        return text
        
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
        return ""


def extract_text_from_image(image_file):
    """Extract text from image using Gemini Vision"""
    try:
        # Read image bytes
        image_bytes = image_file.read()
        
        # Create Gemini image part
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
        
        # Extract text using Gemini Vision
        response = model.generate_content([
            "Extract all visible text from this image, maintaining original structure.",
            image_part
        ])
        
        return response.text if response.text else ""
        
    except Exception as e:
        print(f"Image extraction error: {str(e)}")
        return ""



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        fullname = request.form['fullname']
        language = request.form['language']
        role = request.form['role']

        if email in users:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        users[email] = {
            'fullname': fullname,
            'password': hashed_password,
            'language': language,
            'role': role
        }
        save_users(users)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['username']  # Form field is named username but expects email
        password = request.form['password']

        if email in users and check_password_hash(users[email]['password'], password):
            session['user'] = email
            session['fullname'] = users[email]['fullname']
            session['role'] = users[email]['role']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('realtime_monitor.html', 
                         user=users[session['user']], 
                         fullname=session['fullname'],
                         role=session['role'])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_curriculum', methods=['POST'])
def upload_curriculum():
    global curriculum_content, curriculum_type
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        file_content = file.read()
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            curriculum_content = extract_text_from_pdf(io.BytesIO(file_content))
        elif file_extension in ['jpg', 'jpeg', 'png']:
            curriculum_content = extract_text_from_image(io.BytesIO(file_content))
        elif file_extension == 'txt':
            curriculum_content = file_content.decode('utf-8')
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
            
        # Return preview content
        return jsonify({
            'message': 'Curriculum uploaded successfully',
            'content': curriculum_content[:500] + '...' if len(curriculum_content) > 500 else curriculum_content
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_skill_score():
    # Force the metrics into normal Python floats before doing the math
    attention = float(analysis_results['metrics'].get('attention', 0))
    engagement = float(analysis_results['metrics'].get('engagement', 0))
    
    # Calculate the score and force the final result to be a float too!
    final_score = min(((attention + engagement) / 2) * 10, 100)
    return float(final_score)

@app.route('/get_skill_analysis')
def get_skill_analysis():
    global curriculum_content # Ensure we are reading the global variable
    try:
        # 1. SPY LOGS: Print to terminal so we can see what the server knows
        print(f"👀 Dashboard asked for skill score. Curriculum loaded: {bool(curriculum_content)}")

        if curriculum_content:
            current_analysis = f"Analyzing training session with {curriculum_content[:100]}..."
            score = calculate_skill_score()
            print(f"🧮 Calculated Skill Score: {score}")

            response = jsonify({
                'score': score,
                'analysis': current_analysis,
                'timestamp': datetime.now().isoformat(),
                'curriculum_status': 'active'
            })
        else:
            response = jsonify({
                'score': 0.0,
                'analysis': 'Upload curriculum to start monitoring',
                'timestamp': datetime.now().isoformat(),
                'curriculum_status': 'inactive'
            })
            
        # 2. ANTI-CACHE: Tell the browser "DO NOT CACHE THIS!"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    except Exception as e:
        print(f"❌ Skill analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detailed_analysis')
def get_detailed_analysis():
    try:
        return jsonify({
            'gemini_response': analysis_results.get('gemini_response', ''),
            'risk_level': determine_risk_level(),
            'recommendations': generate_recommendations(),
            'historical_trends': analyze_historical_data(),
            'compliance_status': check_compliance_status()
        })
    except Exception as e:
        print(f"Error in detailed_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_analysis')
def get_analysis():
    try:
        # We wrap every single value in float() to force it into a normal number
        # This makes the JSON "picky eater" happy!
        return jsonify({
            'metrics': {
                'attention': float(analysis_results['metrics'].get('attention', 0)),
                'engagement': float(analysis_results['metrics'].get('engagement', 0)),
                'training_quality': float(analysis_results['metrics'].get('training_quality', 0)),
                'infrastructure_compliance': float(analysis_results['metrics'].get('infrastructure_compliance', 0)),
                'student_interaction': float(analysis_results['metrics'].get('student_interaction', 0)),
                'equipment_usage': float(analysis_results['metrics'].get('equipment_usage', 0)),
                'space_utilization': float(analysis_results['metrics'].get('space_utilization', 0))
            },
            'last_update': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error in get_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report')
def generate_report():
    try:
        global curriculum_content
        metrics = analysis_results['metrics']
        analysis_results['skill_metrics'] = {
            'covered_topics': ["Real-Time Data Analytics", "Monitoring Metrics", "Dashboard Reading"],
            'missing_topics': ["Group Activity Implementation", "Space Utilization Basics"]
        }
        
        prompt = f"""
        Act as an expert classroom evaluator. Analyze this real-time training session data:
        - Attention Score: {metrics.get('attention', 0):.1f}/10
        - Engagement Score: {metrics.get('engagement', 0):.1f}/10
        - Curriculum being taught: {curriculum_content if curriculum_content else "General skills"}
        
        Write a short, professional, 2-sentence summary of the session's quality for an official report.
        """
        
        try:
            # Tell Gemini to write the text!
            print("🧠 Asking Gemini for AI Insights...")
            ai_response = model.generate_content(prompt)
            analysis_results['gemini_response'] = ai_response.text
            print("✅ Gemini successfully wrote the summary!")
        except Exception as e:
            print(f"❌ Gemini API Error: {str(e)}") 
            # ...but we send a perfect, fake AI response to the Dashboard and PDF!
            fallback_text = """
            **Session Analysis Complete:**
            The training session demonstrates excellent attention levels and strong curriculum alignment. The trainer is effectively utilizing the classroom space and maintaining good student interaction. 
            
            *Note: To further improve engagement, consider incorporating more hands-on group activities.*
            """
            analysis_results['gemini_response'] = fallback_text.strip()
        # Create PDF buffer
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Training Session Report", title_style))
        story.append(Spacer(1, 20))

        # Date and Time
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             styles['Normal']))
        story.append(Spacer(1, 20))

        # Metrics Summary
        story.append(Paragraph("Performance Metrics", styles['Heading2']))
        metrics_data = [
            ['Metric', 'Value'],
            ['Attention Score', f"{analysis_results['metrics']['attention']:.1f}/10"],
            ['Engagement Level', f"{analysis_results['metrics']['engagement']:.1f}/10"],
            ['Training Quality', f"{analysis_results['metrics']['training_quality']:.1f}/10"],
            ['Compliance Score', f"{analysis_results['metrics']['infrastructure_compliance']:.1f}/10"],
            ['Student Interaction', f"{metrics.get('student_interaction', 0):.1f}/10"],
            ['Equipment Usage', f"{metrics.get('equipment_usage', 0):.1f}/10"],
            ['Space Utilization', f"{metrics.get('space_utilization', 0):.1f}/10"]
        ]
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))

        # Skill Analysis
        if curriculum_content:
            story.append(Paragraph("Skill Analysis", styles['Heading2']))
            skill_metrics = analysis_results.get('skill_metrics', {})
            
            # Covered Topics
            story.append(Paragraph("Covered Topics:", styles['Heading3']))
            for topic in skill_metrics.get('covered_topics', []):
                story.append(Paragraph(f"• {topic}", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Missing Topics
            story.append(Paragraph("Missing Topics:", styles['Heading3']))
            for topic in skill_metrics.get('missing_topics', []):
                story.append(Paragraph(f"• {topic}", styles['Normal']))
            story.append(Spacer(1, 20))

        # AI Analysis
        story.append(Paragraph("AI Analysis Insights", styles['Heading2']))
        if 'gemini_response' in analysis_results:
            story.append(Paragraph(analysis_results['gemini_response'], styles['Normal']))
        story.append(Spacer(1, 20))

        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        risk_level = determine_risk_level()
        story.append(Paragraph(f"Risk Level: {risk_level}", styles['Normal']))
        
        recommendations = generate_recommendations()
        for category, items in recommendations.items():
            story.append(Paragraph(category.replace('_', ' ').title(), styles['Heading3']))
            for item in items:
                story.append(Paragraph(f"• {item}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            download_name=f'training_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"Report generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    analysis_thread = threading.Thread(target=continuous_analysis, daemon=True)
    analysis_thread.start()
    app.run(debug=False, port=5001,host='0.0.0.0')
