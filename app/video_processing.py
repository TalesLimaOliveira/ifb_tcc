"""
Video Processing Module for LIBRAS Translation

This module handles all video, image, and camera processing operations
including MediaPipe hand landmark detection and OpenCV operations.

Components:
- MediaPipeProcessor: Handles MediaPipe initialization and landmark extraction
- VideoProcessor: Processes video files and camera streams
- ImageProcessor: Handles static image processing
- LandmarkVisualizer: Draws landmarks on images/frames
"""

import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

# Import MediaPipe with error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"MediaPipe not available: {e}")


class MediaPipeProcessor:
    """
    Handles MediaPipe hands detection and landmark extraction.
    
    This class encapsulates MediaPipe functionality for detecting and
    extracting hand landmarks from images and video frames.
    """
    
    def __init__(self, config):
        """
        Initialize MediaPipe processor with configuration.
        
        Args:
            config (dict): MediaPipe configuration parameters
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is not available. Please install it with: pip install mediapipe")
        
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hands detector with error handling
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=config.get("static_image_mode", False),
                max_num_hands=config.get("max_num_hands", 2),
                min_detection_confidence=config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=config.get("min_tracking_confidence", 0.5)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Hands: {e}")
    
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a single frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
        
        Returns:
            tuple: (landmarks_data: np.ndarray, results: MediaPipe results)
                  landmarks_data shape: (max_hands, 21, 3)
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame to detect hands
        results = self.hands.process(frame_rgb)
        
        # Initialize landmarks array for maximum hands
        max_hands = self.config.get("max_num_hands", 2)
        landmarks_data = np.zeros((max_hands, 21, 3))
        
        # Extract landmark coordinates if hands are detected
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx < max_hands:
                    for point_idx, landmark in enumerate(hand_landmarks.landmark):
                        landmarks_data[hand_idx, point_idx] = [
                            landmark.x, 
                            landmark.y, 
                            landmark.z
                        ]
        
        return landmarks_data, results
    
    def has_hands_detected(self, results):
        """
        Check if any hands were detected in the results.
        
        Args:
            results: MediaPipe detection results
        
        Returns:
            bool: True if hands were detected, False otherwise
        """
        return results.multi_hand_landmarks is not None
    
    def get_hand_count(self, results):
        """
        Get the number of hands detected.
        
        Args:
            results: MediaPipe detection results
        
        Returns:
            int: Number of detected hands
        """
        if results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0


class LandmarkVisualizer:
    """
    Handles drawing landmarks and connections on images/frames.
    
    This class provides utilities for visualizing MediaPipe hand landmarks
    and connections on images for debugging and user feedback.
    """
    
    def __init__(self, mp_hands, mp_drawing):
        """
        Initialize the landmark visualizer.
        
        Args:
            mp_hands: MediaPipe hands solution
            mp_drawing: MediaPipe drawing utilities
        """
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
    
    def draw_landmarks(self, frame, results, draw_connections=True):
        """
        Draw hand landmarks and connections on a frame.
        
        Args:
            frame (np.ndarray): Input frame to draw on
            results: MediaPipe detection results
            draw_connections (bool): Whether to draw hand connections
        
        Returns:
            np.ndarray: Frame with landmarks drawn
        """
        # Create a copy to avoid modifying the original frame
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS if draw_connections else None
                )
        
        return annotated_frame
    
    def draw_landmark_info(self, frame, results, show_coordinates=False):
        """
        Draw additional information about detected landmarks.
        
        Args:
            frame (np.ndarray): Input frame
            results: MediaPipe detection results
            show_coordinates (bool): Whether to show coordinate values
        
        Returns:
            np.ndarray: Frame with information overlay
        """
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand index
                cv2.putText(
                    annotated_frame,
                    f"Hand {hand_idx + 1}",
                    (10, 30 + hand_idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                if show_coordinates:
                    # Draw landmark coordinates for specific points (e.g., fingertips)
                    key_points = [4, 8, 12, 16, 20]  # Fingertip landmarks
                    for point_idx in key_points:
                        if point_idx < len(hand_landmarks.landmark):
                            landmark = hand_landmarks.landmark[point_idx]
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            
                            cv2.putText(
                                annotated_frame,
                                f"({x},{y})",
                                (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1
                            )
        
        return annotated_frame


class VideoProcessor:
    """
    Handles video file processing and camera stream operations.
    
    This class manages video capture from files or camera devices,
    frame extraction, and processing coordination with MediaPipe.
    """
    
    def __init__(self, mediapipe_processor, visualizer):
        """
        Initialize video processor.
        
        Args:
            mediapipe_processor (MediaPipeProcessor): MediaPipe handler
            visualizer (LandmarkVisualizer): Landmark visualization handler
        """
        self.mp_processor = mediapipe_processor
        self.visualizer = visualizer
    
    def process_camera_stream(self, camera_index=0):
        """
        Initialize camera capture.
        
        Args:
            camera_index (int): Camera device index
        
        Returns:
            cv2.VideoCapture: Camera capture object or None if failed
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return None
        
        return cap
    
    def process_frame(self, frame, show_landmarks=True, show_info=False):
        """
        Process a single frame for landmark detection and visualization.
        
        Args:
            frame (np.ndarray): Input frame
            show_landmarks (bool): Whether to draw landmarks
            show_info (bool): Whether to show additional information
        
        Returns:
            tuple: (processed_frame, landmarks_data, has_hands)
        """
        # Extract landmarks
        landmarks_data, results = self.mp_processor.extract_landmarks(frame)
        
        # Determine if hands were detected
        has_hands = self.mp_processor.has_hands_detected(results)
        
        # Create output frame
        if show_landmarks and has_hands:
            processed_frame = self.visualizer.draw_landmarks(frame, results)
            if show_info:
                processed_frame = self.visualizer.draw_landmark_info(
                    processed_frame, results
                )
        else:
            processed_frame = frame.copy()
        
        return processed_frame, landmarks_data, has_hands
    
    def process_video_file(self, file_path, max_frames=None, frame_skip=1):
        """
        Process a video file and extract landmarks from frames.
        
        Args:
            file_path (str): Path to video file
            max_frames (int): Maximum number of frames to process
            frame_skip (int): Number of frames to skip between processing
        
        Yields:
            tuple: (frame_number, processed_frame, landmarks_data, has_hands)
        """
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {file_path}")
        
        frame_count = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check frame limits
                if max_frames and processed_count >= max_frames:
                    break
                
                # Skip frames if specified
                if frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                processed_frame, landmarks_data, has_hands = self.process_frame(frame)
                
                yield frame_count, processed_frame, landmarks_data, has_hands
                
                frame_count += 1
                processed_count += 1
        
        finally:
            cap.release()
    
    def save_temporary_video(self, uploaded_file):
        """
        Save uploaded video file to temporary location.
        
        Args:
            uploaded_file: Streamlit uploaded file object
        
        Returns:
            str: Path to temporary file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    
    def cleanup_temporary_file(self, file_path):
        """
        Clean up temporary video file.
        
        Args:
            file_path (str): Path to temporary file
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not cleanup temporary file {file_path}: {e}")


class ImageProcessor:
    """
    Handles static image processing operations.
    
    This class manages image loading, format conversion, and processing
    for landmark detection in static images.
    """
    
    def __init__(self, mediapipe_processor, visualizer):
        """
        Initialize image processor.
        
        Args:
            mediapipe_processor (MediaPipeProcessor): MediaPipe handler
            visualizer (LandmarkVisualizer): Landmark visualization handler
        """
        self.mp_processor = mediapipe_processor
        self.visualizer = visualizer
    
    def load_image_from_file(self, uploaded_file):
        """
        Load image from uploaded file and convert to OpenCV format.
        
        Args:
            uploaded_file: Streamlit uploaded file object
        
        Returns:
            np.ndarray: Image in BGR format for OpenCV processing
        """
        # Load image using PIL
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert to BGR format if needed
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            # RGB to BGR conversion
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            # RGBA to BGR conversion
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:
            # Grayscale or other formats
            image_bgr = image_np
        
        return image_bgr
    
    def process_image(self, image, show_landmarks=True, show_info=False):
        """
        Process a static image for landmark detection.
        
        Args:
            image (np.ndarray): Input image in BGR format
            show_landmarks (bool): Whether to draw landmarks
            show_info (bool): Whether to show additional information
        
        Returns:
            tuple: (processed_image, landmarks_data, has_hands)
        """
        # Extract landmarks
        landmarks_data, results = self.mp_processor.extract_landmarks(image)
        
        # Determine if hands were detected
        has_hands = self.mp_processor.has_hands_detected(results)
        
        # Create output image
        if show_landmarks and has_hands:
            processed_image = self.visualizer.draw_landmarks(image, results)
            if show_info:
                processed_image = self.visualizer.draw_landmark_info(
                    processed_image, results
                )
        else:
            processed_image = image.copy()
        
        return processed_image, landmarks_data, has_hands
    
    def resize_image(self, image, max_width=800, max_height=600):
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image (np.ndarray): Input image
            max_width (int): Maximum width
            max_height (int): Maximum height
        
        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            return resized_image
        
        return image

def get_hands_landmarks(results):
    """Extrai landmarks das mãos e garante um formato padrão (2, 21, 3)."""
    all_hands = []
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            arr = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            all_hands.append(arr)
    
    # Garante que sempre haverá dados para 2 mãos, preenchendo com zeros
    while len(all_hands) < 2:
        all_hands.append(np.zeros((21, 3)))
        
    return np.stack(all_hands) # Shape: (2, 21, 3)

def process_for_prediction(landmarks_data):
    """Processa os landmarks para o formato esperado pelo modelo: (seq_len, 42, 3)."""
    if isinstance(landmarks_data, list):
        landmarks_data = np.array(landmarks_data)

    if landmarks_data.ndim == 3: # Imagem única (2, 21, 3)
        landmarks_data = np.expand_dims(landmarks_data, axis=0)

    # Formato esperado: (seq_len, 2, 21, 3)
    seq_len, _, n_points, n_coords = landmarks_data.shape
    
    # Concatena para o formato final: (seq_len, 42, 3)
    return landmarks_data.reshape(seq_len, 2 * n_points, n_coords)
