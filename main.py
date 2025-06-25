import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import json
from collections import deque
import time

class HandGestureRecognizer:
    def __init__(self, model_path=None, confidence_threshold=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.confidence_threshold = confidence_threshold
        self.gesture_buffer = deque(maxlen=10)
        
        self.gesture_classes = [
            'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down', 
            'peace_sign', 'ok_sign', 'pointing', 'rock_sign'
        ]
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.gesture_classes)
        
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(63,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(self.gesture_classes), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        wrist = np.array(landmarks[0:3])
        normalized_landmarks = []
        
        for i in range(0, len(landmarks), 3):
            point = np.array(landmarks[i:i+3])
            normalized_point = point - wrist
            normalized_landmarks.extend(normalized_point)
        
        return np.array(normalized_landmarks)
    
    def classify_gesture(self, landmarks):
        if self.model is None:
            return "No Model", 0.0
        
        landmarks = landmarks.reshape(1, -1)
        predictions = self.model.predict(landmarks, verbose=0)
        confidence = np.max(predictions)
        
        if confidence > self.confidence_threshold:
            gesture_idx = np.argmax(predictions)
            gesture_name = self.label_encoder.inverse_transform([gesture_idx])[0]
            return gesture_name, confidence
        
        return "Unknown", confidence
    
    def smooth_predictions(self, gesture, confidence):
        self.gesture_buffer.append((gesture, confidence))
        
        if len(self.gesture_buffer) < 5:
            return gesture, confidence
        
        gesture_counts = {}
        
        for g, c in self.gesture_buffer:
            if g not in gesture_counts:
                gesture_counts[g] = []
            gesture_counts[g].append(c)
        
        best_gesture = gesture
        best_confidence = 0
        
        for g, confidences in gesture_counts.items():
            avg_confidence = np.mean(confidences)
            if avg_confidence > best_confidence:
                best_gesture = g
                best_confidence = avg_confidence
        
        return best_gesture, best_confidence
    
    def collect_training_data(self, gesture_name, duration=10):
        cap = cv2.VideoCapture(0)
        landmarks_data = []
        
        print(f"Collecting data for '{gesture_name}' gesture...")
        print(f"Get ready! Collection starts in 3 seconds...")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("START! Show the gesture now!")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    landmarks_data.append(landmarks)
                    
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            remaining = duration - (time.time() - start_time)
            cv2.putText(frame, f"Collecting: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time left: {remaining:.1f}s", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {len(landmarks_data)}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Collected {len(landmarks_data)} samples for '{gesture_name}'")
        return landmarks_data
    
    def train_model(self, training_data_path="gesture_data.pkl"):
        print("Training gesture recognition model...")
        
        if os.path.exists(training_data_path):
            with open(training_data_path, 'rb') as f:
                training_data = pickle.load(f)
        else:
            training_data = self.collect_all_training_data()
            with open(training_data_path, 'wb') as f:
                pickle.dump(training_data, f)
        
        X = []
        y = []
        
        for gesture, landmarks_list in training_data.items():
            for landmarks in landmarks_list:
                X.append(landmarks)
                y.append(gesture)
        
        X = np.array(X)
        y = self.label_encoder.transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return history
    
    def collect_all_training_data(self):
        training_data = {}
        
        for gesture in self.gesture_classes:
            print(f"\nCollecting data for: {gesture}")
            input(f"Press Enter when ready to collect '{gesture}' gesture data...")
            landmarks_data = self.collect_training_data(gesture, duration=15)
            training_data[gesture] = landmarks_data
        
        return training_data
    
    def save_model(self, model_path="gesture_model.h5"):
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def real_time_recognition(self):
        cap = cv2.VideoCapture(0)
        fps_counter = 0
        fps_timer = time.time()
        
        print("Starting real-time gesture recognition...")
        print("Press 'q' to quit, 'c' to collect data, 't' to train model")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self.extract_landmarks(hand_landmarks)
                    gesture, confidence = self.classify_gesture(landmarks)
                    gesture, confidence = self.smooth_predictions(gesture, confidence)
                    
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Hand Gesture Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                gesture_name = input("Enter gesture name to collect: ")
                if gesture_name:
                    landmarks_data = self.collect_training_data(gesture_name)
                    data_file = f"{gesture_name}_data.pkl"
                    with open(data_file, 'wb') as f:
                        pickle.dump(landmarks_data, f)
                    print(f"Data saved to {data_file}")
                cap = cv2.VideoCapture(0)
            elif key == ord('t'):
                cap.release()
                cv2.destroyAllWindows()
                print("Training model...")
                self.train_model()
                self.save_model()
                cap = cv2.VideoCapture(0)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, 0, None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = self.extract_landmarks(hand_landmarks)
                gesture, confidence = self.classify_gesture(landmarks)
                
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                cv2.putText(image, f"{gesture} ({confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                return gesture, confidence, image
        
        return "No hand detected", 0, image


class GestureControlSystem:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.gesture_actions = {}
        self.gesture_history = deque(maxlen=100)
        self.load_gesture_mappings()
    
    def load_gesture_mappings(self):
        config_file = "gesture_mappings.json"
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.gesture_actions = json.load(f)
        else:
            self.gesture_actions = {
                'open_palm': {'action': 'pause_media', 'description': 'Pause/Play media'},
                'closed_fist': {'action': 'click', 'description': 'Mouse click'},
                'thumbs_up': {'action': 'volume_up', 'description': 'Increase volume'},
                'thumbs_down': {'action': 'volume_down', 'description': 'Decrease volume'},
                'peace_sign': {'action': 'scroll_up', 'description': 'Scroll up'},
                'ok_sign': {'action': 'enter', 'description': 'Enter key'},
                'pointing': {'action': 'mouse_move', 'description': 'Move mouse cursor'},
                'rock_sign': {'action': 'screenshot', 'description': 'Take screenshot'}
            }
            
            with open(config_file, 'w') as f:
                json.dump(self.gesture_actions, f, indent=2)
    
    def execute_gesture_action(self, gesture, confidence):
        if gesture in self.gesture_actions and confidence > 0.8:
            action_info = self.gesture_actions[gesture]
            action = action_info['action']
            
            self.gesture_history.append({
                'gesture': gesture,
                'confidence': confidence,
                'action': action,
                'timestamp': time.time()
            })
            
            print(f"Executing action: {action} (gesture: {gesture}, confidence: {confidence:.2f})")
            self.perform_action(action)
    
    def perform_action(self, action):
        print(f"Action performed: {action}")


class DataAugmentation:
    @staticmethod
    def add_noise(landmarks, noise_factor=0.01):
        noise = np.random.normal(0, noise_factor, landmarks.shape)
        return landmarks + noise
    
    @staticmethod
    def scale_landmarks(landmarks, scale_factor_range=(0.8, 1.2)):
        scale_factor = np.random.uniform(*scale_factor_range)
        return landmarks * scale_factor
    
    @staticmethod
    def rotate_landmarks(landmarks, max_angle=15):
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle)
        
        points = landmarks.reshape(-1, 3)[:, :2]
        center = np.mean(points, axis=0)
        
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
        
        rotated_points = np.dot(points - center, rotation_matrix.T) + center
        
        rotated_landmarks = landmarks.copy()
        for i in range(len(rotated_points)):
            rotated_landmarks[i*3:(i*3)+2] = rotated_points[i]
        
        return rotated_landmarks
    
    @staticmethod
    def augment_dataset(landmarks_list, augmentation_factor=3):
        augmented_data = []
        
        for landmarks in landmarks_list:
            augmented_data.append(landmarks)
            
            for _ in range(augmentation_factor):
                aug_landmarks = landmarks.copy()
                
                if np.random.random() > 0.5:
                    aug_landmarks = DataAugmentation.add_noise(aug_landmarks)
                
                if np.random.random() > 0.5:
                    aug_landmarks = DataAugmentation.scale_landmarks(aug_landmarks)
                
                if np.random.random() > 0.5:
                    aug_landmarks = DataAugmentation.rotate_landmarks(aug_landmarks)
                
                augmented_data.append(aug_landmarks)
        
        return augmented_data


def main():
    print("Hand Gesture Recognition System")
    print("=" * 40)
    
    recognizer = HandGestureRecognizer()
    
    while True:
        print("\nOptions:")
        print("1. Real-time recognition")
        print("2. Train model")
        print("3. Process image")
        print("4. Collect training data")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            recognizer.real_time_recognition()
        
        elif choice == '2':
            print("Training model...")
            recognizer.train_model()
            recognizer.save_model()
            print("Model training completed!")
        
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                gesture, confidence, annotated_image = recognizer.process_image(image_path)
                if annotated_image is not None:
                    print(f"Detected gesture: {gesture} (confidence: {confidence:.2f})")
                    cv2.imshow("Gesture Recognition", annotated_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Could not process image")
            else:
                print("Image file not found")
        
        elif choice == '4':
            gesture_name = input("Enter gesture name to collect data for: ").strip()
            if gesture_name:
                landmarks_data = recognizer.collect_training_data(gesture_name)
                data_file = f"{gesture_name}_training_data.pkl"
                with open(data_file, 'wb') as f:
                    pickle.dump(landmarks_data, f)
                print(f"Training data saved to {data_file}")
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()