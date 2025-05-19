import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
import sqlite3
import streamlit as st
import tempfile
import traceback
import base64
from io import BytesIO

# Global variables
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  
MODEL_PATH = "models/plant_disease_model.h5"
PLANT_TYPES = ["Apple", "Cherry", "Corn", "Grape", 
               "Peach", "Pepper", "Potato", 
                "Strawberry", "Tomato"]

st.set_page_config(
    page_title="AI Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar 
st.sidebar.title("🌿 Plant Care Assistant")
st.sidebar.markdown("""
Use the AI-powered detector to diagnose plant diseases from images and get care tips.

**Steps:**
1. Select the plant type
2. Upload a plant leaf image or use camera
3. Get diagnosis and treatment advice
""")

st.sidebar.markdown("---")

# Main header
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #2E8B57;
        text-align: center;
        margin-top: -40px;
        margin-bottom: 20px;
    }
    .sub-text {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #3CB371;
        color: white;
    }
    .camera-container {
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
        position: relative;
    }
    #video-element {
        width: 100%;
        border-radius: 10px;
        border: 2px solid #2E8B57;
    }
    #canvas-element {
        display: none;
    }
    .capture-btn {
        margin-top: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

class PlantDiseaseDetector:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.model = None
        self.class_names = []
        self.current_plant = None
        self.care_tips_data = self._load_care_tips_from_db()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for care tips."""
        conn = sqlite3.connect('plant_care.db')
        cursor = conn.cursor()
        
        # Create care_tips table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS care_tips (
                disease TEXT PRIMARY KEY,
                treatment TEXT,
                prevention TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def _load_care_tips_from_db(self):
        """Load care tips from the database."""
        care_tips_data = []
        try:
            with sqlite3.connect('plant_care.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT disease, treatment, prevention FROM care_tips')
                rows = cursor.fetchall()
                for row in rows:
                    care_tips_data.append(row)
        except Exception as e:
            print(f"Database error: {str(e)}")
        return care_tips_data

    def _get_care_tips(self, disease):
        """Fetch care tips from database."""
        try:
            with sqlite3.connect('plant_care.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT treatment, prevention FROM care_tips WHERE disease=?', (disease,))
                result = cursor.fetchone()
            
            if result:
                return {'treatment': result[0], 'prevention': result[1]}
            else:
                return {
                    'treatment': 'General care: Ensure proper sunlight, water, and nutrients.',
                    'prevention': 'Regularly inspect plants for early signs of disease.'
                }
        except Exception as e:
            print(f"Database error: {str(e)}")
            return {
                'treatment': 'Error retrieving treatment information.',
                'prevention': 'Error retrieving prevention information.'
            }

    def set_plant_type(self, plant_type):
        """Set the current plant type for focused prediction."""
        if plant_type in PLANT_TYPES:
            self.current_plant = plant_type
            print(f"Plant type set to: {plant_type}")
            return True
        else:
            print(f"Invalid plant type: {plant_type}")
            print(f"Available plants: {', '.join(PLANT_TYPES)}")
            return False

    def get_plant_classes(self, plant_type=None):
        """Get class names for a specific plant type or all if None."""
        if not self.class_names:
            self.load_class_names()
            
        if not plant_type:
            return self.class_names
            
        plant_classes = [cls for cls in self.class_names if cls.lower().startswith(plant_type.lower())]
        return plant_classes

    def load_dataset(self):
        """Load and prepare the dataset for training or validation."""
        print("Loading dataset...")
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = valid_datagen.flow_from_directory(
            self.dataset_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        return train_generator, validation_generator

    def build_model(self, num_classes):
        """Use transfer learning with a pre-trained model."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self):
        """Train the model on the dataset."""
        if not self.dataset_path:
            print("Error: Dataset path not specified")
            return
        
        train_generator, validation_generator = self.load_dataset()
        self.model = self.build_model(len(self.class_names))
        print(self.model.summary())
        
        checkpoint = ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        print(f"\nTraining model...")
        history = self.model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[checkpoint, early_stop, reduce_lr]
        )
        
        self.save_class_names()
        self.plot_training_history(history)
        self.evaluate(validation_generator)
        
        return history

    def save_class_names(self):
        """Save class names to a file."""
        os.makedirs('models', exist_ok=True)
        with open("class_names.txt", "r") as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"Class names saved to models/class_names.txt")

    def load_class_names(self):
        """Load class names from file."""
        try:
            with open('models/class_names.txt', 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.class_names)} classes")
        except FileNotFoundError:
            print("Class names file not found")

    def load_trained_model(self):
        """Load a pre-trained model."""
        try:
            self.model = load_model("plant_disease_model.h5")
            self.load_class_names()
            print(f"Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, image_path):
        """Predict disease and return diagnosis + care tips."""
        if not self.model:
           print("Attempting to load model...")
           if not self.load_trained_model():
               print("Failed to load model")
               return None, None, None
           else:
               print("Model loaded successfully")
        
        try:
            print(f"Processing image: {image_path}")
            img = Image.open(image_path)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            print("Making prediction...")
            predictions = self.model.predict(img_array)
            print(f"Raw predictions: {predictions}")
            
            if self.current_plant:
                plant_classes = self.get_plant_classes(self.current_plant)
                if plant_classes:
                    class_indices = {cls: i for i, cls in enumerate(self.class_names)}
                    filtered_predictions = np.zeros_like(predictions[0])
                    for cls in plant_classes:
                        if cls in class_indices:
                            idx = class_indices[cls]
                            filtered_predictions[idx] = predictions[0][idx]
                    
                    if np.sum(filtered_predictions) > 0:
                        predicted_class_index = np.argmax(filtered_predictions)
                        confidence = filtered_predictions[predicted_class_index]
                    else:
                        return f"No valid prediction for {self.current_plant}", 0.0, {"treatment": "", "prevention": ""}
                else:
                    return f"No classes found for {self.current_plant}", 0.0, {"treatment": "", "prevention": ""}
            else:
                predicted_class_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_index]
            
            if predicted_class_index < len(self.class_names):
                predicted_class = self.class_names[predicted_class_index]
                plant_disease = self.parse_class_name(predicted_class)
                tips = self._get_care_tips(predicted_class)
                return plant_disease, confidence, tips
            else:
                return "Unknown", 0.0, {"treatment": "", "prevention": ""}
        except Exception as e:
            print(f"Error predicting image: {str(e)}")
            return None, None, None

    def predict_from_base64(self, base64_image):
        """Predict disease from base64 image data."""
        if not self.model:
            print("Attempting to load model...")
            if not self.load_trained_model():
                print("Failed to load model")
                return None, None, None
            
        try:
            # Decode base64 to image
            image_data = base64.b64decode(base64_image.split(',')[1])
            image = Image.open(BytesIO(image_data))
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                return self.predict(tmp_file.name)
        except Exception as e:
            print(f"Error predicting from base64: {str(e)}")
            print(traceback.format_exc())
            return None, None, None

    def parse_class_name(self, class_name):
        """Parse the class name to extract plant type and disease condition."""
        parts = class_name.split('___') if '___' in class_name else class_name.split('_')
        plant_type = parts[0].replace('_', ' ')
        if len(parts) > 1:
            condition = parts[1].replace('_', ' ')
            return f"{plant_type} - {condition}"
        else:
            if "healthy" in class_name.lower():
                return f"{plant_type} - Healthy"
            else:
                return class_name.replace('_', ' ')

def main():
    detector = PlantDiseaseDetector()
    if not detector.load_trained_model():
        st.error("No trained model found. Please train a model first.")
        return

    st.markdown('<div class="main-title">AI Plant Disease Detector & Care Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Upload a leaf image to detect diseases and get treatment & prevention tips</div>', unsafe_allow_html=True)

    # Add a decorative separator
    st.markdown("<hr style='border: 1px solid #ccc;' />", unsafe_allow_html=True) 
    
    plant_type = st.selectbox("Select Plant Type", [""] + PLANT_TYPES)
    if plant_type:
        detector.set_plant_type(plant_type)

    st.write("### Take a picture or upload an image of a leaf")
    
    tab1, tab2 = st.tabs(["Camera", "Upload"])
    
    with tab1:
        # Custom camera implementation with back camera support
        st.markdown("""
        <div class="camera-container">
            <video id="video-element" autoplay playsinline></video>
            <canvas id="canvas-element"></canvas>
        </div>
        """, unsafe_allow_html=True)
        
        # JavaScript for camera handling
        st.markdown("""
        <script>
            // Wait for DOM to load
            document.addEventListener('DOMContentLoaded', function() {
                const videoElement = document.getElementById('video-element');
                const canvasElement = document.getElementById('canvas-element');
                
                // Set up canvas
                canvasElement.width = 640;
                canvasElement.height = 480;
                
                // Access the back camera by default
                async function startCamera() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: { 
                                facingMode: { exact: "environment" } // This forces the back camera
                            },
                            audio: false
                        });
                        videoElement.srcObject = stream;
                    } catch (err) {
                        // If back camera fails, try the default camera
                        console.error("Back camera not available: ", err);
                        try {
                            const stream = await navigator.mediaDevices.getUserMedia({
                                video: true,
                                audio: false
                            });
                            videoElement.srcObject = stream;
                            console.log("Using default camera");
                        } catch (fallbackErr) {
                            console.error("Camera access failed: ", fallbackErr);
                            // Inform the user
                            const cameraContainer = document.querySelector('.camera-container');
                            cameraContainer.innerHTML = '<div style="color: red; text-align: center;">Camera access denied or not available. Please use the Upload tab.</div>';
                        }
                    }
                }
                
                // Start camera when the page loads
                startCamera();
            });
        </script>
        """, unsafe_allow_html=True)
        
        # Capture button
        capture_btn = st.button("Capture Image")
        
        # Handle the captured image
        if 'camera_image' not in st.session_state:
            st.session_state.camera_image = None
            
        # Let's use a simpler approach with a file uploader
        if capture_btn:
            st.markdown("""
            <div id="capture-result" style="display:none"></div>
            <script>
                // Capture image when button is clicked
                const videoElement = document.getElementById('video-element');
                const canvasElement = document.getElementById('canvas-element');
                const resultElement = document.getElementById('capture-result');
                
                if (videoElement && canvasElement) {
                    // Set canvas dimensions
                    canvasElement.width = videoElement.videoWidth || 640;
                    canvasElement.height = videoElement.videoHeight || 480;
                    
                    // Draw video frame to canvas
                    const context = canvasElement.getContext('2d');
                    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                    
                    // Get base64 image data
                    const imageData = canvasElement.toDataURL('image/jpeg');
                    
                    // Store the image in local storage for retrieval
                    localStorage.setItem('capturedImage', imageData);
                    
                    // Signal that image is ready 
                    resultElement.textContent = "Image captured!";
                    resultElement.style.display = "block";
                    
                    // Force page reload to process the image
                    window.location.reload();
                }
            </script>
            """, unsafe_allow_html=True)
        
        # Check if there's a captured image in localStorage
        captured_image = st.empty()
        st.markdown("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Check if we have a stored image
                const capturedImage = localStorage.getItem('capturedImage');
                if (capturedImage) {
                    // Create an element to display the status
                    const statusElement = document.createElement('div');
                    statusElement.textContent = 'Processing captured image...';
                    statusElement.style.color = '#2E8B57';
                    statusElement.style.padding = '10px';
                    statusElement.style.marginTop = '10px';
                    statusElement.style.textAlign = 'center';
                    
                    // Find the correct container to append to (streamlit's structure)
                    const containers = document.querySelectorAll('.element-container');
                    if (containers.length > 0) {
                        containers[containers.length - 1].appendChild(statusElement);
                    }
                    
                    // Create a form element
                    const form = document.createElement('form');
                    form.method = 'post';
                    form.enctype = 'multipart/form-data';
                    
                    // Create input for the image
                    const input = document.createElement('input');
                    input.type = 'hidden';
                    input.name = 'camera_capture';
                    input.value = capturedImage;
                    
                    // Append and submit
                    form.appendChild(input);
                    document.body.appendChild(form);
                    
                    // Convert base64 to blob for file upload
                    fetch(capturedImage)
                        .then(res => res.blob())
                        .then(blob => {
                            // Create a file from blob
                            const file = new File([blob], "captured_image.jpg", {
                                type: "image/jpeg"
                            });
                            
                            // Create a file list to simulate a file upload
                            const dataTransfer = new DataTransfer();
                            dataTransfer.items.add(file);
                            
                            // Find the file uploader input element
                            const fileInputs = document.querySelectorAll('input[type="file"]');
                            if (fileInputs.length > 0) {
                                // Set the files property
                                fileInputs[0].files = dataTransfer.files;
                                
                                // Trigger a change event
                                const event = new Event('change', { bubbles: true });
                                fileInputs[0].dispatchEvent(event);
                                
                                // Clear the stored image
                                localStorage.removeItem('capturedImage');
                            }
                        });
                }
            });
        </script>
        """, unsafe_allow_html=True)
        
        # Add a hidden file uploader for camera captures
        camera_file = st.file_uploader("Camera Capture", type=["jpg", "png"], key="camera_capture", label_visibility="collapsed")
        
        if camera_file is not None:
            st.image(camera_file, caption="Captured Image", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(camera_file.getvalue())
                image_path = tmp_file.name
                try:
                    predicted_class, confidence, tips = detector.predict(image_path)
                    if predicted_class:
                        st.success(f"Result: {predicted_class} ({confidence*100:.2f}% confidence)")
                        st.markdown("---")
                        st.markdown("### Treatment Tips")
                        st.info(tips['treatment'])

                        st.markdown("### Prevention Advice")
                        st.warning(tips['prevention'])
                    else:
                        st.warning("Could not make a prediction. Please try another image.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check the console for more details.")
                    print(f"Error: {traceback.format_exc()}")
    
    with tab2:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name
                try:
                    predicted_class, confidence, tips = detector.predict(image_path)
                    if predicted_class:
                        st.success(f"Result: {predicted_class} ({confidence*100:.2f}% confidence)")
                        st.markdown("---")
                        st.markdown("### Treatment Tips")
                        st.info(tips['treatment'])

                        st.markdown("### Prevention Advice")
                        st.warning(tips['prevention'])
                    else:
                        st.warning("Could not make a prediction. Please try another image.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check the console for more details.")
                    print(f"Error: {traceback.format_exc()}")

    # Footer
    st.markdown("""
        <br><hr>
        <div style='text-align: center; font-size: 14px;'>
        Developed with ❤️ using Streamlit | 2025
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
