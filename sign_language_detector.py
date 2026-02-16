import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import pickle

class SignLanguageDetector:
    def __init__(self, model_path='model.h5'):
        """Initialize the sign language detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load trained model
        try:
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully")
        except:
            print("No model found. Train a model first.")
            self.model = None
    
    def extract_hand_features(self, hand_landmarks):
        """Extract features from hand landmarks."""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features)
    
    def detect_sign(self, frame):
        """Detect sign language from video frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        detected_sign = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract features and predict
                if self.model:
                    features = self.extract_hand_features(hand_landmarks)
                    features = features.reshape(1, -1)
                    prediction = self.model.predict(features, verbose=0)
                    detected_sign = chr(65 + np.argmax(prediction))  # Convert to letter
        
        return frame, detected_sign
    
    def run_realtime_detection(self):
        """Run real-time sign language detection from webcam."""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, detected_sign = self.detect_sign(frame)
            
            # Display detected sign
            if detected_sign:
                cv2.putText(frame, f'Sign: {detected_sign}', (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run_realtime_detection()
