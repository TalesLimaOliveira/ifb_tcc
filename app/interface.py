"""
Interface de Usuario para Tradução de LIBRAS

Componentes:
- UIManager: Gerenciador principal da interface
- CameraInterface: Interface para webcam
- ImageInterface: Interface para upload de imagens
- VideoInterface: Interface para upload de vídeos
- FeedbackManager: Coleta de feedback do usuário
"""

import streamlit as st
import time
import json
import datetime
import os
from typing import Optional, Tuple, Any


class FeedbackManager:
    """
    Manages user feedback collection and logging for model improvement.
    
    This class handles the collection of user feedback on translation quality
    and saves it for future model retraining and analysis.
    """
    
    def __init__(self, log_directory="../logs"):
        """
        Initialize feedback manager.
        
        Args:
            log_directory (str): Directory to save feedback logs
        """
        self.log_directory = log_directory
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Create log directory if it doesn't exist."""
        app_dir = os.path.dirname(os.path.abspath(__file__))
        full_log_dir = os.path.join(os.path.dirname(app_dir), 'logs')
        os.makedirs(full_log_dir, exist_ok=True)
        self.log_directory = full_log_dir
    
    def save_feedback(self, file_type: str, file_name: str, landmarks: Any, 
                     predicted_text: str, feedback: str) -> str:
        """
        Save user feedback to a JSON log file.
        
        Args:
            file_type (str): Type of input ('image', 'video', 'camera')
            file_name (str): Name of the processed file
            landmarks (Any): Landmark data used for prediction
            predicted_text (str): Text predicted by the model
            feedback (str): User feedback ('correct' or 'incorrect')
        
        Returns:
            str: Path to the saved log file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_directory, f"feedback_{timestamp}.json")
        
        log_data = {
            "timestamp": timestamp,
            "file_type": file_type,
            "file_name": file_name,
            "predicted_text": predicted_text,
            "feedback": feedback,
            "landmarks_shape": str(landmarks.shape) if hasattr(landmarks, 'shape') else "None",
            "landmarks_count": len(landmarks) if landmarks else 0
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return log_file
    
    def render_feedback_buttons(self, file_type: str, file_name: str, 
                               landmarks: Any, predicted_text: str) -> Optional[str]:
        """
        Render feedback buttons and handle user interaction.
        
        Args:
            file_type (str): Type of input
            file_name (str): Name of the processed file
            landmarks (Any): Landmark data
            predicted_text (str): Predicted text
        
        Returns:
            Optional[str]: Feedback result or None
        """
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Correct", key=f"correct_{file_type}_{hash(file_name)}"):
                self.save_feedback(file_type, file_name, landmarks, predicted_text, "correct")
                st.success("Feedback saved: Correct translation!")
                return "correct"
        
        with col2:
            if st.button("Incorrect", key=f"incorrect_{file_type}_{hash(file_name)}"):
                self.save_feedback(file_type, file_name, landmarks, predicted_text, "incorrect")
                st.warning("Feedback saved: Will be used for model improvement!")
                return "incorrect"
        
        return None


class CameraInterface:
    """
    Handles the webcam/camera interface for real-time LIBRAS translation.
    
    This class manages camera input, real-time processing, and display
    of live translation results.
    """
    
    def __init__(self, video_processor, model_manager):
        """
        Initialize camera interface.
        
        Args:
            video_processor: Video processing handler
            model_manager: Neural network model manager
        """
        self.video_processor = video_processor
        self.model_manager = model_manager
    
    def render_controls(self) -> Tuple[bool, bool]:
        """
        Render camera control buttons.
        
        Returns:
            Tuple[bool, bool]: (start_pressed, stop_pressed)
        """
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("Start Camera", key="start_camera")
        with col2:
            stop_button = st.button("Stop Camera", key="stop_camera")
        
        return start_button, stop_button
    
    def process_camera_stream(self, video_placeholder, translation_placeholder,
                            sequence_length: int, show_landmarks: bool, 
                            real_time_translation: bool):
        """
        Process camera stream for real-time translation.
        
        Args:
            video_placeholder: Streamlit placeholder for video display
            translation_placeholder: Streamlit placeholder for translation text
            sequence_length (int): Maximum frames to keep in sequence
            show_landmarks (bool): Whether to show landmark visualization
            real_time_translation (bool): Whether to translate in real-time
        """
        # Initialize camera
        cap = self.video_processor.process_camera_stream()
        
        if cap is None:
            st.error("Error: Could not open camera. Please check camera connection.")
            return
        
        landmarks_sequence = []
        
        try:
            while st.session_state.get('camera_active', False):
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Process frame
                processed_frame, landmarks_data, has_hands = self.video_processor.process_frame(
                    frame, show_landmarks=show_landmarks
                )
                
                # Update landmarks sequence
                if has_hands:
                    landmarks_sequence.append(landmarks_data)
                    
                    # Maintain sequence length
                    if len(landmarks_sequence) > sequence_length:
                        landmarks_sequence.pop(0)
                
                # Display frame
                video_placeholder.image(
                    processed_frame, 
                    channels="BGR", 
                    use_container_width=True,
                    caption="Live Camera Feed"
                )
                
                # Real-time translation (menos frequente para otimização)
                if real_time_translation and len(landmarks_sequence) >= 3:
                    # Fazer predição apenas a cada 10 frames para economizar processamento
                    if len(landmarks_sequence) % 5 == 0:
                        predicted_text = self.model_manager.predict(landmarks_sequence[-10:])  # Usar apenas últimos 10 frames
                        translation_placeholder.markdown(f"**Tradução:** {predicted_text}")
                elif not real_time_translation:
                    translation_placeholder.markdown("**Tradução em tempo real desabilitada**")
                else:
                    translation_placeholder.markdown("**Coletando gestos de mãos...**")
                
                # Delay otimizado
                time.sleep(0.05)
        
        finally:
            cap.release()


class ImageInterface:
    """
    Handles static image upload and processing interface.
    
    This class manages image upload, processing, and feedback collection
    for static image translation.
    """
    
    def __init__(self, image_processor, model_manager, feedback_manager):
        """
        Initialize image interface.
        
        Args:
            image_processor: Image processing handler
            model_manager: Neural network model manager
            feedback_manager: Feedback collection manager
        """
        self.image_processor = image_processor
        self.model_manager = model_manager
        self.feedback_manager = feedback_manager
    
    def render_upload_area(self):
        """
        Render image upload area.
        
        Returns:
            Optional: Uploaded file object or None
        """
        return st.file_uploader(
            "Upload an image with LIBRAS signs",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    def process_uploaded_image(self, uploaded_file, video_placeholder, 
                             translation_placeholder, feedback_placeholder,
                             show_landmarks: bool):
        """
        Process uploaded image and handle results.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            video_placeholder: Placeholder for image display
            translation_placeholder: Placeholder for translation
            feedback_placeholder: Placeholder for feedback buttons
            show_landmarks (bool): Whether to show landmarks
        """
        if uploaded_file is None:
            video_placeholder.info("Please upload an image to start translation")
            return
        
        try:
            # Load and process image
            image = self.image_processor.load_image_from_file(uploaded_file)
            
            # Resize if too large
            image = self.image_processor.resize_image(image)
            
            # Process for landmarks
            processed_image, landmarks_data, has_hands = self.image_processor.process_image(
                image, show_landmarks=show_landmarks
            )
            
            # Display processed image
            video_placeholder.image(
                processed_image,
                channels="BGR",
                use_container_width=True,
                caption=f"Processed Image: {uploaded_file.name}"
            )
            
            # Handle translation and feedback
            if has_hands:
                # Make prediction
                landmarks_sequence = [landmarks_data]
                predicted_text = self.model_manager.predict(landmarks_sequence)
                
                # Display translation
                translation_placeholder.markdown(f"**Translation:** {predicted_text}")
                
                # Render feedback buttons
                with feedback_placeholder.container():
                    st.subheader("Was this translation correct?")
                    self.feedback_manager.render_feedback_buttons(
                        "image", 
                        uploaded_file.name, 
                        landmarks_sequence, 
                        predicted_text
                    )
            else:
                translation_placeholder.warning("No hands detected in the image")
                feedback_placeholder.empty()
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


class VideoInterface:
    """
    Handles video file upload and processing interface.
    
    This class manages video upload, frame-by-frame processing,
    and feedback collection for video translation.
    """
    
    def __init__(self, video_processor, model_manager, feedback_manager):
        """
        Initialize video interface.
        
        Args:
            video_processor: Video processing handler
            model_manager: Neural network model manager
            feedback_manager: Feedback collection manager
        """
        self.video_processor = video_processor
        self.model_manager = model_manager
        self.feedback_manager = feedback_manager
    
    def render_upload_area(self):
        """
        Render video upload area.
        
        Returns:
            Optional: Uploaded file object or None
        """
        return st.file_uploader(
            "Upload a video with LIBRAS signs",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
    
    def process_uploaded_video(self, uploaded_file, video_placeholder,
                             translation_placeholder, feedback_placeholder,
                             max_frames: int, show_landmarks: bool):
        """
        Process uploaded video and handle results.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            video_placeholder: Placeholder for video display
            translation_placeholder: Placeholder for translation
            feedback_placeholder: Placeholder for feedback
            max_frames (int): Maximum frames to process
            show_landmarks (bool): Whether to show landmarks
        """
        if uploaded_file is None:
            video_placeholder.info("Please upload a video to start translation")
            return
        
        # Process button
        if not st.button("Process Video", key="process_video_btn"):
            return
        
        # Save temporary file
        temp_path = None
        try:
            temp_path = self.video_processor.save_temporary_video(uploaded_file)
            
            # Initialize processing
            landmarks_sequence = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video frames with optimization
            frame_count = 0
            from config import UI_CONFIG
            frame_skip = UI_CONFIG.get("frame_skip", 2)
            
            for frame_num, processed_frame, landmarks_data, has_hands in \
                self.video_processor.process_video_file(temp_path, max_frames, frame_skip):
                
                # Update progress
                progress = min(frame_count / max_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processando frame {frame_num + 1}...")
                
                # Display current frame (menos frequente para otimização)
                if frame_count % 3 == 0:  # Mostrar apenas alguns frames
                    video_placeholder.image(
                        processed_frame,
                        channels="BGR",
                        use_container_width=True,
                        caption=f"Frame {frame_num + 1}"
                    )
                
                # Collect landmarks only from frames with hands
                if has_hands:
                    landmarks_sequence.append(landmarks_data)
                
                frame_count += 1
                
                # Smaller delay for better responsiveness
                time.sleep(0.01)
            
            # Final processing
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Generate translation
            if landmarks_sequence:
                predicted_text = self.model_manager.predict(landmarks_sequence)
                translation_placeholder.markdown(f"**Translation:** {predicted_text}")
                
                # Render feedback buttons
                with feedback_placeholder.container():
                    st.subheader("Was this translation correct?")
                    self.feedback_manager.render_feedback_buttons(
                        "video",
                        uploaded_file.name,
                        landmarks_sequence,
                        predicted_text
                    )
            else:
                translation_placeholder.warning("No hands detected in the video")
                feedback_placeholder.empty()
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        
        finally:
            # Cleanup
            if temp_path:
                self.video_processor.cleanup_temporary_file(temp_path)


class UIManager:
    """
    Main UI manager that coordinates all interface components.
    
    This class orchestrates the entire user interface, manages navigation,
    and coordinates between different processing modules.
    """
    
    def __init__(self, model_manager, video_processor, image_processor):
        """
        Initialize UI manager.
        
        Args:
            model_manager: Neural network model manager
            video_processor: Video processing handler
            image_processor: Image processing handler
        """
        self.model_manager = model_manager
        self.video_processor = video_processor
        self.image_processor = image_processor
        self.feedback_manager = FeedbackManager()
        
        # Initialize interface components
        self.camera_interface = CameraInterface(video_processor, model_manager)
        self.image_interface = ImageInterface(image_processor, model_manager, self.feedback_manager)
        self.video_interface = VideoInterface(video_processor, model_manager, self.feedback_manager)
    
    def render_header(self):
        """Render application header and title."""
        # Remove duplicate title since it's already shown in main()
        # Model status indicator only shown if there's an error
        model_info = self.model_manager.get_model_info()
        if model_info["status"] != "Model loaded":
            st.error("Neural model not loaded")
    
    def render_sidebar(self) -> dict:
        """
        Render sidebar with configuration options.
        
        Returns:
            dict: Configuration parameters selected by user
        """
        st.sidebar.title("Configuration")
        
        # Input type selection
        input_type = st.sidebar.selectbox(
            "Choose input type:",
            ["Camera/Webcam", "Image Upload", "Video Upload"],
            help="Select the type of input for LIBRAS translation"
        )
        
        # Advanced configuration
        with st.sidebar.expander("Advanced Settings"):
            config = {
                "confidence_threshold": st.slider(
                    "Minimum detection confidence", 
                    0.1, 1.0, 0.5, 0.1,
                    help="Minimum confidence for hand detection"
                ),
                "max_frames": st.number_input(
                    "Maximum frames for video", 
                    10, 200, 50,
                    help="Maximum number of frames to process from video"
                ),
                "sequence_length": st.number_input(
                    "Temporal sequence length", 
                    5, 100, 30,
                    help="Number of frames to use for translation"
                ),
                "show_landmarks": st.checkbox(
                    "Show hand landmarks", 
                    value=True,
                    help="Display detected hand landmarks on video/image"
                ),
                "real_time_translation": st.checkbox(
                    "Real-time translation", 
                    value=True,
                    help="Enable continuous translation for camera input"
                )
            }
        
        config["input_type"] = input_type
        return config
    
    def render_main_interface(self, config: dict):
        """
        Render main application interface based on configuration.
        
        Args:
            config (dict): User configuration parameters
        """
        # Create main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Processing")
            video_placeholder = st.empty()
        
        with col2:
            st.subheader("Translation Result")
            translation_placeholder = st.empty()
            
            # Only show feedback for image/video uploads
            if config["input_type"] in ["Image Upload", "Video Upload"]:
                feedback_placeholder = st.empty()
            else:
                feedback_placeholder = None
        
        # Route to appropriate interface based on input type
        if config["input_type"] == "Camera/Webcam":
            self._handle_camera_interface(
                config, video_placeholder, translation_placeholder, feedback_placeholder
            )
        elif config["input_type"] == "Image Upload":
            self._handle_image_interface(
                config, video_placeholder, translation_placeholder, feedback_placeholder
            )
        elif config["input_type"] == "Video Upload":
            self._handle_video_interface(
                config, video_placeholder, translation_placeholder, feedback_placeholder
            )
    
    def _handle_camera_interface(self, config, video_placeholder, 
                                translation_placeholder, feedback_placeholder):
        """Handle camera interface logic."""
        # Show appropriate button based on camera state
        camera_active = st.session_state.get('camera_active', False)
        
        if camera_active:
            # Show stop button above the camera
            if st.button("Stop Camera", key="stop_camera"):
                st.session_state.camera_active = False
                st.rerun()
            
            # Process camera stream
            self.camera_interface.process_camera_stream(
                video_placeholder,
                translation_placeholder,
                config["sequence_length"],
                config["show_landmarks"],
                config["real_time_translation"]
            )
        else:
            # Show start button and status
            if st.button("Start Camera", key="start_camera"):
                st.session_state.camera_active = True
                st.rerun()
            
            video_placeholder.info("Click 'Start Camera' to begin real-time translation")
    
    def _handle_image_interface(self, config, video_placeholder,
                               translation_placeholder, feedback_placeholder):
        """Handle image interface logic."""
        uploaded_file = self.image_interface.render_upload_area()
        self.image_interface.process_uploaded_image(
            uploaded_file,
            video_placeholder,
            translation_placeholder,
            feedback_placeholder,
            config["show_landmarks"]
        )
    
    def _handle_video_interface(self, config, video_placeholder,
                               translation_placeholder, feedback_placeholder):
        """Handle video interface logic."""
        uploaded_file = self.video_interface.render_upload_area()
        self.video_interface.process_uploaded_video(
            uploaded_file,
            video_placeholder,
            translation_placeholder,
            feedback_placeholder,
            config["max_frames"],
            config["show_landmarks"]
        )
    
    def render_footer(self):
        """Render application footer with information."""
        # Model information only if needed for debugging
        model_info = self.model_manager.get_model_info()
        if model_info["status"] != "Model loaded":
            st.sidebar.markdown("---")
            st.sidebar.error("Model not loaded properly")
    
    def run(self):
        """Run the complete application interface."""
        # Initialize session state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        # Render components
        self.render_header()
        config = self.render_sidebar()
        self.render_main_interface(config)
        self.render_footer()
