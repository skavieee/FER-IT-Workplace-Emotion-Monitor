from flask import Flask, Response
import cv2
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)

# Load your CLCM model
model = tf.keras.models.load_model('clcm_model.h5')

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Simple Haar Cascade face detection (NO MediaPipe needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Preprocess for CLCM model (224x224)
                face_resized = cv2.resize(face_roi, (224, 224))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                face_input = np.stack((face_gray,)*3, axis=-1).astype(np.float32) / 255.0
                
                # Predict emotion
                pred = model.predict(np.expand_dims(face_input, 0), verbose=0)[0]
                emotion = EMOTIONS[np.argmax(pred)]
                confidence = np.max(pred) * 100
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(frame, f'{emotion}: {confidence:.1f}%', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = base64.b64encode(buffer).decode()
        yield (b'data: image/jpeg;base64,' + frame_bytes.encode() + b'\n\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>🎭 FER - IEEE CLCM Demo</title>
    <style>
        body { font-family: Arial; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 0; padding: 20px; }
        h1 { font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
        #video { width: 640px; height: 480px; border: 5px solid rgba(255,255,255,0.3); border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .info { background: rgba(0,0,0,0.3); padding: 20px; margin: 20px auto; border-radius: 10px; max-width: 640px; }
    </style>
    </head>
    <body>
        <h1>🎓 Facial Emotion Recognition</h1>
        <h2>IEEE CLCM Model Demo (2.3M params)</h2>
        <img id="video" src="/video_feed" alt="Live Feed">
        <div class="info">
            <p><strong>📊 Domain:</strong> Computer Vision + Machine Learning</p>
            <p><strong>🔬 Base Paper:</strong> IEEE Access 2024 [Gursesli et al.]</p>
            <p><strong>⚡ Status:</strong> Real-time webcam processing</p>
        </div>
        <script>
        const video = document.getElementById("video");
        const evtSource = new EventSource("/video_feed");
        evtSource.onmessage = function(event) {
            video.src = "data:image/jpeg;base64," + event.data;
        };
        </script>
    </body></html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🚀 Starting FER Demo at http://localhost:5000")
    print("✅ TensorFlow warnings are NORMAL (performance optimization)")
    app.run(debug=False, host='0.0.0.0', port=5000)
