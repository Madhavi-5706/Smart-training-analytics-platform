import google.generativeai as genai
import cv2
import numpy as np
from datetime import datetime
import os
import threading
import requests
import time
import re
from dotenv import load_dotenv

load_dotenv()

class SkillMonitor:
    def __init__(self, curriculum_path):
        self.curriculum_path = curriculum_path
        self.curriculum_content = self._load_curriculum()
        
        # Set up Gemini API with proper SSL context and timeout
        import ssl
        import socket
        import urllib3
        urllib3.disable_warnings()
        
        # Configure longer timeout
        socket.setdefaulttimeout(30)
        ssl._create_default_https_context = ssl._create_unverified_context
        
        genai.configure(api_key="AIzaSyAMYpivJ4uwYaQdbPoRCPy-qXqAf0G8w1A")
        self.genai = genai.GenerativeModel('gemini-1.5-flash')
        
        # IP webcam configuration
        self.ip_address = "192.168.1.101:8080"
        
        # Add locks for file access
        self._video_lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        
        # Temporary file management
        self.temp_dir = "temp_videos"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        self.skill_analysis_results = {
            'timestamp': None,
            'analysis': {
                'scores': {
                    'skill_coverage': 0,
                    'practical_implementation': 0,
                    'sequence_adherence': 0,
                    'time_management': 0,
                    'learning_objectives': 0
                },
                'overall_match': 0,
                'recommendations': []
            }
        }
        self.recording_active = False
        self.analysis_thread = None
        
    def _load_curriculum(self):
        with open(self.curriculum_path, 'r') as file:
            return file.read()
    
    def start_monitoring(self):
        self.recording_active = True
        self.analysis_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self.analysis_thread.start()

    def stop_monitoring(self):
        self.recording_active = False
        if self.analysis_thread:
            self.analysis_thread.join()

    def record_temp_video(self, duration=40):
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                url = f"http://{self.ip_address}/shot.jpg"
                img_resp = requests.get(url, timeout=5)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                frame = cv2.imdecode(img_arr, -1)
                
                if frame is not None and frame.size > 0:
                    frames.append(frame)
                    
                time.sleep(1)  # 1 FPS recording
                
            except Exception as e:
                print(f"Frame capture error: {str(e)}")
                continue
                
        return frames

    def _save_temp_video(self, frames, output_path):
        if not frames:
            return False
            
        try:
            with self._video_lock:
                full_path = os.path.join(self.temp_dir, output_path)
                # Ensure the file doesn't exist before writing
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                    except:
                        # If can't remove, use a new filename
                        base, ext = os.path.splitext(output_path)
                        output_path = f"{base}_{int(time.time())}{ext}"
                        full_path = os.path.join(self.temp_dir, output_path)
                    
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(full_path, fourcc, 1, (width, height))
                
                for frame in frames:
                    out.write(frame)
                out.release()
                cv2.destroyAllWindows()  # Ensure all windows are closed
                return full_path
                
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            return False

    def _compare_with_curriculum(self, video_path):
        with self._analysis_lock:
            try:
                if not os.path.exists(video_path):
                    print("Video file not found")
                    return None
                
                # Retry logic for Gemini API
                max_retries = 3
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        media_file = genai.upload_file(video_path)
                        response = self.genai.generate_content(
                            [self._get_analysis_prompt(), media_file],
                            timeout=30
                        )
                        return self._parse_analysis(response.text)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise e
                
            except Exception as e:
                print(f"Gemini analysis error: {str(e)}")
                return None
            finally:
                # Clean up the video file with retry logic
                max_cleanup_attempts = 3
                for attempt in range(max_cleanup_attempts):
                    try:
                        if os.path.exists(video_path):
                            cv2.destroyAllWindows()  # Ensure all windows are closed
                            time.sleep(0.5)  # Give time for resources to be released
                            os.remove(video_path)
                            break
                    except Exception as e:
                        if attempt == max_cleanup_attempts - 1:
                            print(f"Failed to clean up video file after {max_cleanup_attempts} attempts: {str(e)}")
                        time.sleep(1)

    def _get_analysis_prompt(self):
        return f"""
        Analyze this training video segment against the following curriculum and provide detailed scoring:

        CURRICULUM CONTENT:
        {self.curriculum_content}

        ANALYSIS REQUIREMENTS:
        1. Skill Coverage (Score 0-10)
        - How well does the session cover curriculum topics?
        - Are all required skills being addressed?
        
        2. Practical Implementation (Score 0-10)
        - Quality of skill demonstrations
        - Correctness of techniques shown
        
        3. Sequence Adherence (Score 0-10)
        - Does training follow curriculum order?
        - Logical progression of concepts
        
        4. Time Management (Score 0-10)
        - Appropriate time allocation per topic
        - Pace of instruction
        
        5. Learning Objectives Achievement (Score 0-10)
        - Evidence of meeting curriculum goals
        - Student comprehension indicators

        Provide:
        1. Numerical scores for each category (format exactly as "Category: X.X")
        2. Brief explanation for each score
        3. Overall match percentage
        4. 2-3 specific recommendations for improvement
        5. Highlight any critical gaps

        Format the response clearly with scores and explanations.
        """

    def _parse_analysis(self, response_text):
        analysis = {
            'scores': {
                'skill_coverage': self._extract_score(response_text, 'Skill Coverage'),
                'practical_implementation': self._extract_score(response_text, 'Practical Implementation'),
                'sequence_adherence': self._extract_score(response_text, 'Sequence Adherence'),
                'time_management': self._extract_score(response_text, 'Time Management'),
                'learning_objectives': self._extract_score(response_text, 'Learning Objectives')
            },
            'recommendations': self._extract_recommendations(response_text),
            'timestamp': datetime.now()
        }
        
        # Calculate overall match
        scores = list(analysis['scores'].values())
        analysis['overall_match'] = sum(scores) / len(scores) * 10
        
        return analysis

    def _extract_score(self, text, category):
        try:
            # Look for patterns like "Category: 8.5" or "Category (Score: 8.5)"
            patterns = [
                f"{category}:?\s*(\d+\.?\d*)",
                f"{category}.*?score:?\s*(\d+\.?\d*)",
                f"{category}.*?(\d+\.?\d*)/10"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            return 0
        except:
            return 0

    def _extract_recommendations(self, text):
        try:
            # Look for recommendations section
            recommendations = []
            rec_section = re.search(r"recommendations?:?(.*?)(?:\n\n|$)", 
                                  text, 
                                  re.IGNORECASE | re.DOTALL)
            
            if rec_section:
                # Split into individual recommendations
                items = rec_section.group(1).split('\n')
                recommendations = [item.strip('- ').strip() 
                                 for item in items 
                                 if item.strip('- ').strip()]
                
            return recommendations[:5]  # Return top 5 recommendations
        except:
            return []

    def _continuous_monitoring(self):
        while self.recording_active:
            try:
                frames = self.record_temp_video()
                if frames:
                    temp_video = f"temp_skill_analysis_{int(time.time())}.mp4"
                    saved_path = self._save_temp_video(frames, temp_video)
                    if saved_path:
                        analysis = self._compare_with_curriculum(saved_path)
                        
                        if analysis:
                            self.skill_analysis_results = {
                                'timestamp': datetime.now(),
                                'analysis': analysis
                            }
                
                time.sleep(40)  # Wait before next analysis cycle
                
            except Exception as e:
                print(f"Monitoring cycle error: {str(e)}")
                time.sleep(5)  # Short delay on error before retry

    def get_latest_analysis(self):
        return self.skill_analysis_results
