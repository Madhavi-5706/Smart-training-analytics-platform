from flask import Flask, render_template, Response, jsonify , request
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
from email.mime.multipart import MIMEMultipart

load_dotenv()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variables
# Add at the top with other global variables
# Email Configuration
# Email Configuration
SENDER_EMAIL = "bhavanisaladi9182@gmail.com"
SENDER_PASSWORD = "zhco sqvv vyvf hoav"
RECEIVER_EMAIL = "gudisasandeep141312@gmail.com"


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

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')
text_model = genai.GenerativeModel('gemini-1.5-pro')
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
                detector_backend='retinaface'
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
                    send_distraction_alert(engagement_state, attention_score)
                
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
def send_distraction_alert(student_state, attention_score):
    try:
        # Create email message
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = RECEIVER_EMAIL
        message["Subject"] = f"Student Attention Alert - {student_state}"
        
        # Create email body
        body = f"""
        Attention Alert!
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Student State: {student_state}
        Attention Score: {attention_score:.2f}/10
        
        This is an automated alert from the Training Monitoring System.
        Please check on the student's engagement level.
        """
        
        message.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
            print(f"Alert email sent for {student_state} state")
            
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
    frames_buffer = []
    last_video_time = datetime.now()
    camera_ip = "192.0.0.4:8080"
    
    while True:
        try:
            url = f"http://{camera_ip}/video"
            stream = requests.get(url, stream=True, timeout=1)
            bytes_data = bytes()
            
            for chunk in stream.iter_content(chunk_size=16384):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
        
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame = cv2.resize(frame, (640, 480))
                        frames_buffer.append(frame.copy())
                        
                        metrics = process_frame(frame, metrics_only=True)
                        if metrics:
                            analysis_results['metrics'].update(metrics)
                        
                        current_time = datetime.now()
                        if (current_time - last_video_time > timedelta(seconds=30) and len(frames_buffer) > 0):
                            print("\n=== Recording video for Gemini Analysis ===")
                            video_path = "temp_video.mp4"
                            
                            height, width = frames_buffer[0].shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(video_path, fourcc, 1, (width, height))
                            
                            for frame in frames_buffer:
                                out.write(frame)
                            
                            out.release()
                            frames_buffer = []
                            last_video_time = current_time
                            
                            print("=== Sending to Gemini for Analysis ===")
                            try:
                                with open(video_path, 'rb') as f:
                                    media_file = genai.upload_file(video_path)
                                    response = model.generate_content([ANALYSIS_PROMPT, media_file])
                                    print(response.text)
                                    
                                    if response and hasattr(response, 'text'):
                                        cleaned_response = clean_text(response.text)
                                        analysis_results['gemini_response'] = cleaned_response
                                        analysis_results['last_update'] = current_time
                                        print("\nGemini Analysis Updated:", cleaned_response[:100])
                                        
                                        session_metrics['attention_avg'].append(metrics['attention'])
                                        session_metrics['engagement_avg'].append(metrics['engagement'])
                                        session_metrics['training_quality_avg'].append(metrics['training_quality'])
                                        session_metrics['gemini_insights'].append({
                                            'timestamp': current_time.isoformat(),
                                            'analysis': cleaned_response
                                        })
                            except Exception as e:
                                print(f"Gemini API error: {str(e)}")
                            
                            if os.path.exists(video_path):
                                os.remove(video_path)
                        
                        if len(frames_buffer) > 30:
                            frames_buffer = frames_buffer[-30:]
                
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Stream error: {str(e)}")
            time.sleep(0.1)

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

def check_compliance_status():
    metrics = analysis_results['metrics']
    compliance_criteria = {
        'infrastructure': {
            'space_utilization': 7.0,
            'equipment_usage': 8.0,
            'safety_standards': 8.5
        },
        'training_quality': {
            'engagement': 7.0,
            'attention': 7.0,
            'interaction': 7.5
        },
        'course_delivery': {
            'training_quality': 7.5,
            'student_interaction': 7.0
        }
    }
    
    compliance_status = {
        'infrastructure': {
            'status': 'compliant' if metrics['infrastructure_compliance'] >= compliance_criteria['infrastructure']['space_utilization'] else 'non_compliant',
            'score': metrics['infrastructure_compliance'],
            'issues': []
        },
        'training_quality': {
            'status': 'compliant' if metrics['training_quality'] >= compliance_criteria['training_quality']['engagement'] else 'non_compliant',
            'score': metrics['training_quality'],
            'issues': []
        },
        'course_delivery': {
            'status': 'compliant' if metrics['student_interaction'] >= compliance_criteria['course_delivery']['student_interaction'] else 'non_compliant',
            'score': metrics['student_interaction'],
            'issues': []
        }
    }
    
    if metrics['space_utilization'] < compliance_criteria['infrastructure']['space_utilization']:
        compliance_status['infrastructure']['issues'].append('Space utilization below required standards')
    
    compliant_categories = sum(1 for category in compliance_status.values() if category['status'] == 'compliant')
    overall_status = 'COMPLIANT' if compliant_categories == len(compliance_status) else 'NON-COMPLIANT'
    
    return {
        'overall_status': overall_status,
        'detailed_status': compliance_status,
        'timestamp': datetime.now().isoformat(),
        'action_required': overall_status == 'NON-COMPLIANT'
    }

def determine_risk_level():
    metrics = analysis_results['metrics']
    avg_score = sum(metrics.values()) / len(metrics)
    
    if avg_score >= 7.5:
        return 'GREEN'
    elif avg_score >= 5.0:
        return 'YELLOW'
    else:
        return 'RED'

def generate_recommendations():
    return {
        'immediate_actions': [],
        'improvement_areas': [],
        'best_practices': []
    }

def generate_frames():
    while True:
        try:
            #ip_address = "192.168.43.1:8080"
            ip_address = "192.0.0.4:8080"
            url = f"http://{ip_address}/shot.jpg"
            
            img_resp = requests.get(url, timeout=5)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                # Process frame for visualization
                frame = process_frame(frame)  # This will draw bounding boxes and metrics
                
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
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


@app.route('/')
def index():
    return render_template('realtime_monitor1.html')

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

# Initialize global variables at the top
session_metrics1 = {
    'attention_avg': [],
    'engagement_avg': [],
    'training_quality_avg': [],
    'gemini_insights': [],
    'skill_monitoring': []
}
@app.route('/get_skill_analysis')
def get_skill_analysis():
    try:
        if curriculum_content:
            # Generate analysis based on current metrics
            current_analysis = f"Analyzing training session with {curriculum_content[:100]}..."
            score = calculate_skill_score()
            
            return jsonify({
                'score': score,
                'analysis': current_analysis,
                'timestamp': datetime.now().isoformat(),
                'curriculum_status': 'active'
            })
        return jsonify({
            'score': 0,
            'analysis': 'Upload curriculum to start monitoring',
            'timestamp': datetime.now().isoformat(),
            'curriculum_status': 'inactive'
        })
    except Exception as e:
        print(f"Skill analysis error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

def calculate_skill_score():
    # Calculate skill score based on current metrics
    attention = analysis_results['metrics']['attention']
    engagement = analysis_results['metrics']['engagement']
    return min(((attention + engagement) / 2) * 10, 100)
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
        return jsonify({
            'metrics': {
                'attention': analysis_results['metrics'].get('attention', 0),
                'engagement': analysis_results['metrics'].get('engagement', 0),
                'training_quality': analysis_results['metrics'].get('training_quality', 0),
                'infrastructure_compliance': analysis_results['metrics'].get('infrastructure_compliance', 0),
                'student_interaction': analysis_results['metrics'].get('student_interaction', 0),
                'equipment_usage': analysis_results['metrics'].get('equipment_usage', 0),
                'space_utilization': analysis_results['metrics'].get('space_utilization', 0)
            },
            'last_update': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error in get_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    analysis_thread = threading.Thread(target=continuous_analysis, daemon=True)
    analysis_thread.start()
    app.run(debug=False, port=5001,host='0.0.0.0')
