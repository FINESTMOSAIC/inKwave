from flask import Flask, render_template, Response, send_file
import cv2
import mediapipe as mp  
import numpy as np
import os

app = Flask(__name__)
SAVE_PATH = 'saved_drawings'

# Create save path if it doesn't exist
os.makedirs(SAVE_PATH, exist_ok=True)

class HandDetector:
    def __init__(self):
        self.hand_recognition_enabled = True

    def disableHandRecognition(self):
        self.hand_recognition_enabled = False

    def enableHandRecognition(self):
        self.hand_recognition_enabled = True

detector = HandDetector()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands




@app.route('/')
def index():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = None  # Disable camera-related functionality
    if cap:
        global width
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        global height
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a black canvas
        global canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        global selected_color
        selected_color = (0, 255, 0)  # Default to green

    return render_template('index.html')

def generate_frames():
    global selected_color, canvas
    
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and detector.hand_recognition_enabled:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x = int(index_finger_tip.x * width)
                    y = int(index_finger_tip.y * height)
                    cv2.circle(canvas, (x, y), 10, selected_color, -1)

            combined = cv2.addWeighted(frame, 1, canvas, 0.5, 0)
            ret, buffer = cv2.imencode('.jpg', combined)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_black_canvas_frames():
    global canvas
    
    while True:
        # Create a frame with the canvas (black background with colored drawing)
        black_canvas_frame = np.copy(canvas)

        ret, buffer = cv2.imencode('.jpg', black_canvas_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/black_canvas_feed')
def black_canvas_feed():
    return Response(generate_black_canvas_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color/<color>')
def set_color(color):
    global selected_color
    colors = {
        'r': (0, 0, 255),   # Red
        'g': (0, 255, 0),   # Green
        'b': (255, 0, 0),   # Blue
        'e': (0, 0, 0),     # Eraser (black)
        'y': (255, 255, 255), # White
        'z': (254, 50, 185), # Custom Pink
        'f': (124, 134, 0)   # Custom Green
    }
    selected_color = colors.get(color, (0, 255, 0))
    return '', 204

@app.route('/save_canvas')
def save_canvas():
    global canvas
    filename = os.path.join(SAVE_PATH, 'drawing.png')
    cv2.imwrite(filename, canvas)
    return '', 204

@app.route('/show_drawing')
def show_drawing():
    filename = os.path.join(SAVE_PATH, 'drawing.png')
    return send_file(filename, mimetype='image/png')

@app.route('/clear_canvas')
def clear_canvas():
    global canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    return '', 204

@app.route('/toggle_hand_recognition/<status>')
def toggle_hand_recognition(status):
    if status == 'enable':
        detector.enableHandRecognition()
        new_status = 'enabled'
    elif status == 'disable':
        detector.disableHandRecognition()
        new_status = 'disabled'
    else:
        # Handle invalid status
        return 'Invalid status', 400
    
    return new_status, 200


@app.route('/about')
def about():
    return render_template('about.html')

# if __name__ == "__main__":
    # app.run(debug=True)
