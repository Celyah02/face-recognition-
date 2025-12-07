# Face Recognition with MediaPipe

A real-time face recognition system that uses MediaPipe for face detection and OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer for identification. This project allows you to capture face images, train a recognition model, and perform real-time face recognition from your webcam.

## 🎯 Features

- **Face Detection**: Uses MediaPipe FaceMesh for accurate face detection
- **Face Recognition**: Implements LBPH face recognizer for person identification
- **Real-time Processing**: Live video feed with face recognition overlay
- **Easy Workflow**: Simple three-step process (Record → Train → Predict)
- **Automatic Organization**: Organizes training data by person name

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam (camera index 1 by default)
- OpenCV with face module support
- MediaPipe library

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-recognition-mediapipe
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python opencv-contrib-python mediapipe numpy
   ```

   **Note**: For OpenCV face recognition, you need `opencv-contrib-python` which includes the face module. If you already have `opencv-python`, uninstall it first:
   ```bash
   pip uninstall opencv-python
   pip install opencv-contrib-python
   ```

## 📖 How It Works

This project follows a three-step workflow:

### 1. **Record** (`record.py`)
   Captures face images from your webcam using MediaPipe for face detection. Images are automatically cropped to the face region and saved to the dataset.

### 2. **Train** (`train.py`)
   Trains an LBPH face recognizer on the collected dataset. Creates a model file and label mapping for recognition.

### 3. **Predict** (`predict.py`)
   Performs real-time face recognition using the trained model. Displays the recognized person's name and confidence score.

## 🎮 Usage

### Step 1: Record Face Data

Run the recording script to capture face images:

```bash
python record.py
```

- Enter your name when prompted
- Position your face in front of the camera
- The script will automatically capture and save face images
- Press `q` to stop recording
- Images are saved to `dataset/{your_name}/`

**Tips**:
- Capture images in different lighting conditions
- Vary your facial expressions
- Aim for 50-100 images per person for better accuracy

### Step 2: Train the Model

After collecting face data, train the recognition model:

```bash
python train.py
```

This will:
- Load all images from the `dataset/` directory
- Train the LBPH face recognizer
- Save the model to `models/lbph_model.xml`
- Create a label map at `models/label_map.json`

**Note**: Make sure you have at least one person's data in the `dataset/` folder before training.

### Step 3: Run Face Recognition

Start real-time face recognition:

```bash
python predict.py
```

- The webcam feed will open
- Detected faces will be highlighted with a green rectangle
- Recognized names and confidence scores will be displayed
- Press `q` to quit

## 📁 Project Structure

```
face-recognition-mediapipe/
│
├── record.py          # Capture face images for training
├── train.py           # Train the face recognition model
├── predict.py         # Real-time face recognition
├── README.md          # This file
│
├── dataset/           # Training images (created automatically)
│   ├── person1/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └── person2/
│       └── ...
│
└── models/            # Trained models (created automatically)
    ├── lbph_model.xml      # Trained LBPH model
    └── label_map.json      # Person name to label mapping
```

## 🔧 Configuration

### Camera Index

By default, the scripts use camera index `1`. If your webcam is on a different index, modify the following line in `record.py` and `predict.py`:

```python
cap = cv2.VideoCapture(1)  # Change 1 to 0 or another index
```

### Detection Confidence

Adjust face detection sensitivity in `record.py` and `predict.py`:

```python
with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,  # Lower = more sensitive
    min_tracking_confidence=0.5     # Lower = more sensitive
) as fm:
```

## 🧠 Technical Details

### Face Detection
- **MediaPipe FaceMesh**: Detects facial landmarks and provides bounding box coordinates
- Handles face tracking and provides stable detection

### Face Recognition
- **LBPH (Local Binary Patterns Histograms)**: A robust face recognition algorithm
- Works well with varying lighting conditions
- Fast and efficient for real-time applications

### Workflow
1. MediaPipe detects faces and provides bounding boxes
2. Face regions are cropped from the video frame
3. Cropped faces are converted to grayscale
4. LBPH recognizer predicts the person's identity
5. Results are displayed with confidence scores

## ⚠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2.face'"
**Solution**: Install `opencv-contrib-python` instead of `opencv-python`:
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Issue: Camera not working
**Solution**: 
- Check if your camera is being used by another application
- Try changing the camera index (0, 1, 2, etc.)
- Verify camera permissions on your system

### Issue: "Unknown" always displayed
**Solution**:
- Ensure you've trained the model with `train.py`
- Check that `models/lbph_model.xml` exists
- Verify the person's face data is in the dataset
- Try capturing more training images with better lighting

### Issue: Low recognition accuracy
**Solution**:
- Capture more training images (50-100 per person)
- Ensure good lighting conditions during recording
- Capture images with different angles and expressions
- Retrain the model after adding more data

## 📝 Notes

- The confidence score in the prediction is a distance metric (lower is better)
- For best results, ensure consistent lighting between training and prediction
- The system works best with frontal face views
- Multiple people can be added to the dataset by running `record.py` with different names

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available for personal and educational use.

---

**Happy Face Recognizing! 🎭**
