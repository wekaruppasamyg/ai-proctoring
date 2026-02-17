


import cv2
import os
import mediapipe as mp

def capture_and_save_face(save_path="faces/verify_face.jpg"):
    """
    Opens webcam, detects a face, saves the face image to the given save_path (e.g., faces/{username}/verify_face.jpg).
    User can press 'Q' to quit. Returns True if a face is detected and saved, else False.
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return False

    import time
    start_time = time.time()
    timeout = 10  # seconds
    detected = False

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    looking_away_count = 0
    looking_away_threshold = 15  # number of frames

    def is_looking_away(landmarks, img_w, img_h):
        # Use key landmarks: nose tip (1), left eye (33), right eye (263), chin (152)
        # Calculate horizontal nose position relative to eyes
        try:
            nose = landmarks[1]
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            chin = landmarks[152]
            nose_x = nose.x * img_w
            left_x = left_eye.x * img_w
            right_x = right_eye.x * img_w
            # If nose is too close to left or right eye, head is turned
            if nose_x < left_x or nose_x > right_x:
                return True
            # Optionally, check vertical angle (chin vs nose)
            return False
        except Exception:
            return False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        # Head pose detection with mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if is_looking_away(face_landmarks.landmark, frame.shape[1], frame.shape[0]):
                    looking_away_count += 1
                else:
                    looking_away_count = 0
                if looking_away_count > looking_away_threshold:
                    cv2.putText(frame, "Please look at the screen!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                break

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, face_img)
                print(f"Face saved to {save_path}")
                detected = True
                break
            break

        cv2.imshow("Show your face and press Q", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time > timeout:
            print("No face detected within timeout.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return detected
