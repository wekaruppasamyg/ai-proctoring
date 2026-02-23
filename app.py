from flask import Flask, render_template, request, redirect, session, url_for, Response, jsonify
import os
import psycopg2
import random
import string
import base64
import shutil
import subprocess
import tempfile
import sys
from datetime import datetime
import cv2
import numpy as np
from time import time
import threading
from werkzeug.utils import secure_filename
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# ================= APP CONFIG =================
app = Flask(__name__)
app.secret_key = "exam-secret"
os.makedirs("faces", exist_ok=True)

# ================= MJPEG STREAMING =================
streams = {}

def gen_mjpeg(username):
    while True:
        frame = streams.get(username)
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            import time; time.sleep(0.1)

# Endpoint for candidate to POST webcam frames
@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    if 'username' not in session:
        return '', 401
    username = session['username']
    img_data = request.data
    streams[username] = img_data
    return '', 204

# MJPEG stream endpoint for admin to view candidate
@app.route('/mjpeg/<username>')
def mjpeg_stream(username):
    return Response(gen_mjpeg(username), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= DATABASE INIT =================

_db_initialized = False
_db_init_lock = threading.Lock()


def _normalize_database_url(db_url: str) -> str:
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]

    parsed = urlparse(db_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("sslmode", os.environ.get("PGSSLMODE", "require"))
    # Fail fast if DB is slow/unreachable, so requests return JSON errors instead of timing out.
    query.setdefault("connect_timeout", os.environ.get("PGCONNECT_TIMEOUT", "5"))
    return urlunparse(parsed._replace(query=urlencode(query)))

def get_db_connection():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return psycopg2.connect(_normalize_database_url(db_url))

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Camera events table for timestamped logs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS camera_events (
        id SERIAL PRIMARY KEY,
        username TEXT,
        event_type TEXT,
        event_time TEXT,
        exam_subject TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id SERIAL PRIMARY KEY,
        name TEXT,
        email TEXT,
        username TEXT,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        id SERIAL PRIMARY KEY,
        name TEXT,
        enabled BOOLEAN DEFAULT TRUE
    )
    """)

    # Backward compatible schema updates
    cur.execute("ALTER TABLE subjects ADD COLUMN IF NOT EXISTS enabled BOOLEAN DEFAULT TRUE")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id SERIAL PRIMARY KEY,
        subject_id INTEGER,
        question TEXT,
        opt1 TEXT,
        opt2 TEXT,
        opt3 TEXT,
        opt4 TEXT,
        correct INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id SERIAL PRIMARY KEY,
        username TEXT,
        subject TEXT,
        score INTEGER,
        date TIMESTAMPTZ,
        cheating_count INTEGER DEFAULT 0,
        terminated BOOLEAN DEFAULT FALSE,
        looking_away_count INTEGER DEFAULT 0,
        tab_switch_count INTEGER DEFAULT 0,
        camera_hidden_count INTEGER DEFAULT 0,
        hand_cover_count INTEGER DEFAULT 0,
        no_blink_count INTEGER DEFAULT 0
    )
    """)

    cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS camera_hidden_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS hand_cover_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS no_blink_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS head_movement_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS eye_tracker_count INTEGER DEFAULT 0")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS materials (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        subject_id INTEGER,
        upload_date TIMESTAMPTZ,
        enabled BOOLEAN DEFAULT TRUE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS coding_results (
        id SERIAL PRIMARY KEY,
        username TEXT,
        language TEXT,
        code TEXT,
        output TEXT,
        submitted_at TIMESTAMPTZ,
        looking_away_count INTEGER DEFAULT 0,
        tab_switch_count INTEGER DEFAULT 0,
        camera_hidden_count INTEGER DEFAULT 0,
        hand_cover_count INTEGER DEFAULT 0,
        cheating_count INTEGER DEFAULT 0,
        terminated BOOLEAN DEFAULT FALSE
    )
    """)

    conn.commit()
    conn.close()


def ensure_db_initialized():
    global _db_initialized
    if _db_initialized:
        return
    with _db_init_lock:
        if _db_initialized:
            return
        init_db()
        _db_initialized = True


@app.before_request
def _ensure_db_initialized_before_request():
    ensure_db_initialized()

# ================= HELPERS =================
def generate_username(name):
    return name.lower().replace(" ", "") + str(random.randint(1000, 9999))

def generate_password():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))

def verify_face(username, live_img_path):
    stored_path = f"faces/{username}/face.jpg"
    if not os.path.exists(stored_path):
        return False

    img1 = cv2.imread(stored_path, 0)
    img2 = cv2.imread(live_img_path, 0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


    f1 = face_cascade.detectMultiScale(img1, 1.3, 5)
    f2 = face_cascade.detectMultiScale(img2, 1.3, 5)

    if len(f1) == 0 or len(f2) == 0:
        return "No face detected. Please ensure your face is visible to the camera."

    # Use the first detected face for comparison, ignore if multiple faces
    x, y, w, h = f1[0]
    img1 = cv2.resize(img1[y:y+h, x:x+w], (200, 200))
    x, y, w, h = f2[0]
    img2 = cv2.resize(img2[y:y+h, x:x+w], (200, 200))

    diff = cv2.absdiff(img1, img2)
    score = np.mean(diff)

    print("Face diff score:", score)
    return score < 60

def save_exam_face(username, img):
    folder = f"faces/{username}"
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if f.startswith("exam_")]
    count = len(files) + 1
    path = f"{folder}/exam_{count}.jpg"
    if img is None or not hasattr(img, 'shape'):
        print(f"[ERROR] Image for user {username} is invalid and cannot be saved.")
        return False
    success = cv2.imwrite(path, img)
    if not success:
        print(f"[ERROR] Failed to save image to {path}. Check write permissions and disk space.")
        return False
    print(f"[INFO] Face image saved to {path}")
    return True

def save_registration_face(username, img):
    folder = f"faces/{username}"
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/face.jpg"
    if img is None or not hasattr(img, 'shape'):
        print(f"[ERROR] Registration image for user {username} is invalid and cannot be saved.")
        return False
    success = cv2.imwrite(path, img)
    if not success:
        print(f"[ERROR] Failed to save registration image to {path}.")
        return False
    print(f"[INFO] Registration face image saved to {path}")
    return True

def log_faces_during_exam(session_id):
    """Log faces detected at the start of the exam"""
    pass

# ================= HOME =================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/ai-login")
def ai_login():
    return render_template("ai_login.html")

# ================= STUDENT FLOW =================
# 1. Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]

        username = generate_username(name)
        password = generate_password()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users(name,email,username,password) VALUES(%s,%s,%s,%s)",
            (name, email, username, password)
        )
        conn.commit()
        conn.close()

        session["username"] = username
        session["reg_user"] = username
        session["reg_pass"] = password

        return redirect(url_for("face_register"))

    return render_template("register.html")


# 2. Face Register
@app.route("/face_register", methods=["GET", "POST"])
def face_register():
    if "username" not in session:
        return redirect(url_for("student_login"))

    if request.method == "POST":
        data = request.get_json()
        img_data = data["image"].split(",")[1]

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(img_data), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Save registration face image in user's folder
        save_registration_face(session['username'], img)
        return {"status": "success"}

    return render_template("face_register.html")


# 3. Show Credentials
@app.route("/credentials")
def credentials():
    if "reg_user" not in session:
        return redirect(url_for("student_login"))

    return render_template(
        "credentials.html",
        username=session.pop("reg_user"),
        password=session.pop("reg_pass")
    )


# 4. Student Login Page
@app.route("/student-login")
def student_login():
    return render_template("login.html")


# 5. Login Check
@app.route("/login", methods=["POST"])
def login_check():
    u = request.form["username"]
    p = request.form["password"]

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username=%s AND password=%s",
        (u, p)
    )
    user = cur.fetchone()
    conn.close()


    if user:
        session["username"] = u
        # Face verification removed: go directly to dashboard or next step
        return redirect(url_for("student_dashboard"))

    return "Login Failed"





# 8. Student Dashboard
@app.route("/student-dashboard")
def student_dashboard():
    if "username" not in session:
        return redirect(url_for("student_login"))

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT name,email,username FROM users WHERE username=%s", (session["username"],))
    student = cur.fetchone()

    cur.execute("SELECT id, name, enabled FROM subjects")
    subjects = cur.fetchall()

    conn.close()

    return render_template("student_dashboard.html", student=student, subjects=subjects)

# ================= VIEW RESULTS =================
@app.route("/view-results")
def view_results():
    if "username" not in session:
        return redirect(url_for("student_login"))

    username = session["username"]

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT subject, score, date
        FROM results
        WHERE username=%s
        ORDER BY date DESC
        """,
        (username,),
    )
    results = cur.fetchall()
    conn.close()

    return render_template("admin_results.html", results=results)

# 9. Start Exam
@app.route("/start-exam", methods=["POST"])
def start_exam():
    if "username" not in session:
        return redirect(url_for("student_login"))

    # This must match the form name
    subject_id = request.form["subject_id"]

    # Log all faces detected at the start of the exam
    log_faces_during_exam(session_id=session["username"])

    conn = get_db_connection()
    cur = conn.cursor()

    # Get subject name
    cur.execute("SELECT name, enabled FROM subjects WHERE id=%s", (subject_id,))
    subject_row = cur.fetchone()
    if not subject_row:
        return "Subject not found"
    subject_name, enabled = subject_row
    if not enabled:
        return "This subject is currently locked. Please contact admin."

    # Get questions
    cur.execute(
        """
        SELECT id, question, opt1, opt2, opt3, opt4
        FROM questions
        WHERE subject_id=%s
        """,
        (subject_id,),
    )
    questions = cur.fetchall()
    conn.close()

    if not questions:
        return f"<h2>No questions found for {subject_name}</h2>"

    # Initialize cheating and event tracking
    session['cheating_count'] = 0
    session['terminated'] = False
    session['looking_away_count'] = 0
    session['tab_switch_count'] = 0
    session['camera_hidden_count'] = 0
    session['hand_cover_count'] = 0
    session['last_look_away_ts'] = 0  # reset debounce timer
    session['last_camera_hidden_ts'] = 0
    session['last_hand_cover_ts'] = 0
    session['last_eye_tracker_ts'] = 0
    session['last_head_movement_ts'] = 0
    session['eye_tracker_count'] = 0
    session['head_movement_count'] = 0
    # Blink tracking (only track no-blink events)
    session['no_blink_count'] = 0
    session['last_no_blink_ts'] = 0
    # Store subject for later use
    session['current_subject'] = subject_name
    return render_template("exam.html", questions=questions, subject=subject_name)

@app.route("/monitor-exam", methods=["POST"])
def monitor_exam():
    if "username" not in session:
        return {"status": "unauthorized"}

    if session.get('terminated', False):
        return {"status": "terminated"}

    print("ðŸ“¸ Monitor exam API called")

    data = request.get_json()
    LOOK_AWAY_COOLDOWN = 3  # seconds
    CAMERA_HIDDEN_COOLDOWN = 5  # seconds
    HAND_COVER_COOLDOWN = 4  # seconds

    def increment_look_away():
        now = time()
        last = session.get("last_look_away_ts", 0)
        if now - last >= LOOK_AWAY_COOLDOWN:
            session['looking_away_count'] = session.get('looking_away_count', 0) + 1
            session['last_look_away_ts'] = now
        return {"status": "look-away", "looking_away_count": session['looking_away_count']}

    def increment_camera_hidden():
        now = time()
        last = session.get("last_camera_hidden_ts", 0)
        if now - last >= CAMERA_HIDDEN_COOLDOWN:
            session['camera_hidden_count'] = session.get('camera_hidden_count', 0) + 1
            session['last_camera_hidden_ts'] = now
            # Log event in camera_events table
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO camera_events (username, event_type, event_time, exam_subject) VALUES (%s, %s, NOW(), %s)",
                    (
                        session.get("username", "unknown"),
                        "camera_hidden",
                        session.get("current_subject", "unknown"),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                print("[Camera Event Log Error]", e)
        return {"status": "camera-hidden", "camera_hidden_count": session['camera_hidden_count']}

    def increment_hand_cover():
        now = time()
        last = session.get("last_hand_cover_ts", 0)
        if now - last >= HAND_COVER_COOLDOWN:
            session['hand_cover_count'] = session.get('hand_cover_count', 0) + 1
            session['last_hand_cover_ts'] = now
        return {"status": "hand-cover", "hand_cover_count": session['hand_cover_count']}

    # No-blink handling with cooldown
    def increment_no_blink():
        now = time()
        # don't increment no_blink too frequently; use 5s cooldown
        if now - session.get('last_no_blink_ts', 0) >= 5.0:
            session['no_blink_count'] = session.get('no_blink_count', 0) + 1
            session['last_no_blink_ts'] = now
        return {"status": "no_blink", "no_blink_count": session['no_blink_count']}

    def increment_eye_tracker():
        now = time()
        if now - session.get('last_eye_tracker_ts', 0) >= 2.5:
            session['eye_tracker_count'] = session.get('eye_tracker_count', 0) + 1
            session['last_eye_tracker_ts'] = now
            # Log event in camera_events table
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO camera_events (username, event_type, event_time, exam_subject) VALUES (%s, %s, NOW(), %s)",
                    (
                        session.get("username", "unknown"),
                        "eye_tracker_violation",
                        session.get("current_subject", "unknown"),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                print("[Eye Tracker Event Log Error]", e)
        return {"status": "eye_tracker", "eye_tracker_count": session['eye_tracker_count']}

    def increment_head_movement():
        now = time()
        if now - session.get('last_head_movement_ts', 0) >= 4.0:
            session['head_movement_count'] = session.get('head_movement_count', 0) + 1
            session['last_head_movement_ts'] = now
            # Log event in camera_events table
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO camera_events (username, event_type, event_time, exam_subject) VALUES (%s, %s, NOW(), %s)",
                    (
                        session.get("username", "unknown"),
                        "head_movement_violation",
                        session.get("current_subject", "unknown"),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                print("[Head Movement Event Log Error]", e)
        return {"status": "head_movement", "head_movement_count": session['head_movement_count']}

    if data.get("event") == "looking_away" and data.get("image") is None:
        return increment_look_away()

    if data.get("event") == "camera_hidden":
        return increment_camera_hidden()

    if data.get("event") == "hand_cover":
        return increment_hand_cover()

    if data.get("event") == "no_blink":
        return increment_no_blink()

    if data.get("event") == "eye_tracker_violation":
        return increment_eye_tracker()

    if data.get("event") == "head_movement_violation":
        return increment_head_movement()


    img_data = data["image"].split(",")[1] if data.get("image") else None
    if img_data:
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(img_data), np.uint8),
            cv2.IMREAD_COLOR
        )
        # Save latest snapshot for admin live monitoring
        username = session.get("username")
        if username:
            live_dir = os.path.join("static", "live")
            os.makedirs(live_dir, exist_ok=True)
            live_path = os.path.join(live_dir, f"{username}.jpg")
            cv2.imwrite(live_path, img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(60, 60)
        )
        print("Faces detected:", len(faces))
        if len(faces) == 0:
            return increment_look_away()
        if len(faces) > 1:
            session['cheating_count'] = session.get('cheating_count', 0) + 1
            if session['cheating_count'] >= 10:
                session['terminated'] = True
                return {"status": "terminated", "cheating_count": session['cheating_count']}
            return {"status": "cheating", "cheating_count": session['cheating_count']}
        return {"status": "ok"}
    return {"status": "ok"}


# Admin live monitoring route
@app.route("/admin-live-monitor")
def admin_live_monitor():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    live_dir = os.path.join("static", "live")
    candidates = []
    if os.path.exists(live_dir):
        for fname in os.listdir(live_dir):
            if fname.endswith(".jpg"):
                username = fname[:-4]
                fpath = os.path.join(live_dir, fname)
                ts = int(os.path.getmtime(fpath))
                last_updated = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                candidates.append({
                    "username": username,
                    "last_updated": last_updated,
                    "timestamp": ts
                })
    return render_template("admin_live_monitor.html", candidates=candidates)

# Endpoint to increment tab switch count from frontend
@app.route("/tab-switch", methods=["POST"])
def tab_switch():
    if "username" not in session:
        return {"status": "unauthorized"}
    session['tab_switch_count'] = session.get('tab_switch_count', 0) + 1
    return {"status": "ok", "tab_switch_count": session['tab_switch_count']}



# 10. Submit Exam
@app.route("/submit-exam", methods=["POST"])
def submit_exam():
    if "username" not in session:
        return redirect(url_for("student_login"))

    subject = request.form["subject"]
    username = session["username"]
    score = 0

    cheating_count = session.get('cheating_count', 0)
    terminated = session.get('terminated', False)
    looking_away_count = session.get('looking_away_count', 0)
    tab_switch_count = session.get('tab_switch_count', 0)
    camera_hidden_count = session.get('camera_hidden_count', 0)
    hand_cover_count = session.get('hand_cover_count', 0)
    no_blink_count = session.get('no_blink_count', 0)
    eye_tracker_count = session.get('eye_tracker_count', 0)
    head_movement_count = session.get('head_movement_count', 0)

    if terminated:
        score = 0  # Or handle differently

    conn = get_db_connection()
    cur = conn.cursor()

    total_questions = 0
    for qid in request.form:
        if qid == "subject":
            continue

        cur.execute("SELECT correct FROM questions WHERE id=%s", (qid,))
        correct = cur.fetchone()[0]

        if int(request.form[qid]) == correct:
            score += 1
        total_questions += 1

    cur.execute(
        """
        INSERT INTO results(
            username,subject,score,date,cheating_count,terminated,
            looking_away_count,tab_switch_count,camera_hidden_count,hand_cover_count,no_blink_count,
            head_movement_count,eye_tracker_count
        )
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            username,
            subject,
            score,
            datetime.now(),
            cheating_count,
            terminated,
            looking_away_count,
            tab_switch_count,
            camera_hidden_count,
            hand_cover_count,
            no_blink_count,
            head_movement_count,
            eye_tracker_count,
        ),
    )

    conn.commit()
    conn.close()

    return render_template("exam_finished.html", score=score)


# ================= ADMIN =================
ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form["username"] == ADMIN_USER and request.form["password"] == ADMIN_PASS:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        return "Invalid Admin Login"
    return render_template("admin_login.html")


@app.route("/admin-dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, enabled FROM subjects")
    subjects = cur.fetchall()
    # For each subject, get its questions and correct answers
    subject_data = []
    for subject in subjects:
        subject_id, subject_name, enabled = subject
        cur.execute(
            "SELECT id, question, opt1, opt2, opt3, opt4, correct FROM questions WHERE subject_id=%s",
            (subject_id,)
        )
        questions = cur.fetchall()
        # Prepare questions with correct answer text
        question_list = []
        for q in questions:
            qid, qtext, opt1, opt2, opt3, opt4, correct = q
            options = [opt1, opt2, opt3, opt4]
            correct_answer = options[correct-1] if correct and 1 <= correct <= 4 else "N/A"
            question_list.append({
                'id': qid,
                'question': qtext,
                'options': options,
                'correct': correct,
                'correct_answer': correct_answer
            })
        subject_data.append({
            'id': subject_id,
            'name': subject_name,
            'enabled': enabled,
            'questions': question_list
        })
    conn.close()
    return render_template("admin_dashboard.html", subject_data=subject_data)

# Toggle subject enabled/disabled
@app.route("/toggle-subject/<int:subject_id>", methods=["POST"])
def toggle_subject(subject_id):
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE subjects SET enabled = NOT enabled WHERE id = %s", (subject_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

# Delete subject route
@app.route("/delete-subject", methods=["POST"])
def delete_subject():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    subject_id = request.form["subject_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM subjects WHERE id=%s", (subject_id,))
    cur.execute("DELETE FROM questions WHERE subject_id=%s", (subject_id,))  # Also delete related questions
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

# Delete question route
@app.route("/delete-question", methods=["POST"])
def delete_question():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    question_id = request.form["question_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM questions WHERE id=%s", (question_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_dashboard"))

# Delete all candidates route
@app.route("/delete-all-candidates", methods=["POST"])
def delete_all_candidates():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    # Only delete live images and clear streams, keep database intact
    live_dir = os.path.join("static", "live")
    if os.path.exists(live_dir):
        for fname in os.listdir(live_dir):
            fpath = os.path.join(live_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
    # Clear streams
    streams.clear()
    return redirect(url_for("admin_live_monitor"))

@app.route("/add-subjects", methods=["GET", "POST"])
def add_subjects():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()

    if request.method == "POST":
        cur.execute("INSERT INTO subjects(name) VALUES(%s)", (request.form["name"],))
        conn.commit()

    cur.execute("SELECT * FROM subjects")
    subjects = cur.fetchall()
    conn.close()

    return render_template("add_subjects.html", subjects=subjects)


@app.route("/add-questions", methods=["GET", "POST"])
def add_questions():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM subjects")
    subjects = cur.fetchall()

    if request.method == "POST":
        cur.execute(
            """
            INSERT INTO questions(subject_id,question,opt1,opt2,opt3,opt4,correct)
            VALUES(%s,%s,%s,%s,%s,%s,%s)
            """,
            (
            request.form["subject_id"],
            request.form["question"],
            request.form["opt1"],
            request.form["opt2"],
            request.form["opt3"],
            request.form["opt4"],
            request.form["correct"]
            ),
        )
        conn.commit()

    cur.execute("SELECT * FROM questions")
    questions = cur.fetchall()
    conn.close()
    return render_template("add_question.html", subjects=subjects, questions=questions)

 # ================= ADMIN VIEW ALL RESULTS =================
@app.route("/admin-results")
def admin_results():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT username, subject, score, date, id, cheating_count, terminated, looking_away_count, tab_switch_count, camera_hidden_count, hand_cover_count, no_blink_count
        FROM results
        ORDER BY date DESC
    """)
    results = cur.fetchall()
    conn.close()

    return render_template("admin_results.html", results=results)

@app.route("/delete-result", methods=["POST"])
def delete_result():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    result_id = request.form["result_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM results WHERE id=%s", (result_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_results"))

@app.route("/admin-logout")
def admin_logout():
    session.clear()
    return redirect(url_for("home"))

# Admin view for camera events
@app.route("/admin-camera-events")
def admin_camera_events():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, event_type, event_time, exam_subject FROM camera_events ORDER BY event_time DESC LIMIT 200")
    events = cur.fetchall()
    conn.close()
    return render_template("admin_camera_events.html", events=events)

# Admin manage materials
@app.route("/manage-materials", methods=["GET", "POST"])
def manage_materials():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    
    if request.method == "POST":
        title = request.form["title"]
        description = request.form.get("description", "")
        file = request.files.get("file")
        
        if file and file.filename:
            # Save file to static/materials directory
            materials_dir = os.path.join(app.root_path, "static", "materials")
            os.makedirs(materials_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(materials_dir, filename)
            file.save(file_path)
            
            # Save to database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO materials (title, description, filename, filepath, upload_date) VALUES (%s, %s, %s, %s, %s)",
                (title, description, filename, f"static/materials/{filename}", datetime.now()),
            )
            conn.commit()
            conn.close()
        
        return redirect(url_for("manage_materials"))
    
    # GET request - show all materials
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, description, filename, filepath, upload_date, enabled FROM materials ORDER BY upload_date DESC")
    materials = cur.fetchall()
    conn.close()
    
    return render_template("manage_materials.html", materials=materials)

# Delete material
@app.route("/delete-material", methods=["POST"])
def delete_material():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    
    material_id = request.form["material_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get filename before deleting
    cur.execute("SELECT filename FROM materials WHERE id=%s", (material_id,))
    result = cur.fetchone()
    if result:
        filename = result[0]
        # Delete file from filesystem
        file_path = os.path.join(app.root_path, "static", "materials", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        cur.execute("DELETE FROM materials WHERE id=%s", (material_id,))
        conn.commit()
    
    conn.close()
    return redirect(url_for("manage_materials"))

# Toggle material enabled/disabled status
@app.route("/toggle-material", methods=["POST"])
def toggle_material():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    
    material_id = request.form["material_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get current enabled status and toggle it
    cur.execute("SELECT enabled FROM materials WHERE id=%s", (material_id,))
    result = cur.fetchone()
    if result:
        current_status = result[0]
        new_status = not bool(current_status)
        cur.execute("UPDATE materials SET enabled=%s WHERE id=%s", (new_status, material_id))
        conn.commit()
    
    conn.close()
    return redirect(url_for("manage_materials"))

# Student view materials
@app.route("/materials")
def materials():
    if not session.get("username"):
        return redirect(url_for("student_login"))
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, description, filename, filepath, upload_date FROM materials WHERE enabled = TRUE ORDER BY upload_date DESC"
    )
    materials = cur.fetchall()
    conn.close()
    
    return render_template("materials.html", materials=materials)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("student_login"))

# ================= CODING EXAM MODULE =================

CODING_LANGUAGES = ["Python", "HTML", "JavaScript", "PHP"]

@app.route("/coding-exam", methods=["GET", "POST"])
def coding_exam():
    if "username" not in session:
        return redirect(url_for("student_login"))

    if request.method == "POST":
        language = request.form.get("language", "Python")
        session['coding_language'] = language
        # Reset monitoring counters for coding exam
        session['cheating_count'] = 0
        session['terminated'] = False
        session['looking_away_count'] = 0
        session['tab_switch_count'] = 0
        session['camera_hidden_count'] = 0
        session['hand_cover_count'] = 0
        session['last_look_away_ts'] = 0
        session['last_camera_hidden_ts'] = 0
        session['last_hand_cover_ts'] = 0
        session['last_eye_tracker_ts'] = 0
        session['last_head_movement_ts'] = 0
        session['eye_tracker_count'] = 0
        session['head_movement_count'] = 0
        session['no_blink_count'] = 0
        session['last_no_blink_ts'] = 0
        session['current_subject'] = f"Coding-{language}"
        return render_template("coding_exam.html", language=language, username=session["username"])

    return render_template("coding_exam_select.html", languages=CODING_LANGUAGES)


@app.route("/run-code", methods=["POST"])
def run_code():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    language = data.get("language", "Python")
    code = data.get("code", "")
    output = ""
    error = ""

    try:
        if language == "Python":
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp_path = f.name
            try:
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True, text=True, timeout=10
                )
                output = result.stdout
                error = result.stderr
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        elif language == "JavaScript":
            # Return JS for browser execution in sandboxed iframe
            return jsonify({"type": "browser", "language": "JavaScript", "code": code})

        elif language == "HTML":
            # Return HTML for browser rendering in sandboxed iframe
            return jsonify({"type": "browser", "language": "HTML", "code": code})

        elif language == "PHP":
            with tempfile.NamedTemporaryFile(mode="w", suffix=".php", delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp_path = f.name
            try:
                result = subprocess.run(
                    ["php", tmp_path],
                    capture_output=True, text=True, timeout=10
                )
                output = result.stdout
                error = result.stderr
            except FileNotFoundError:
                error = "PHP is not installed on the server. Code saved for review."
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if error and not output:
            return jsonify({"type": "console", "output": error, "is_error": True})
        combined = output
        if error:
            combined += "\n--- STDERR ---\n" + error
        return jsonify({"type": "console", "output": combined, "is_error": bool(error)})

    except subprocess.TimeoutExpired:
        return jsonify({"type": "console", "output": "⏱ Execution timed out (10s limit).", "is_error": True})
    except Exception as e:
        return jsonify({"type": "console", "output": f"Error: {str(e)}", "is_error": True})


@app.route("/submit-coding-exam", methods=["POST"])
def submit_coding_exam():
    if "username" not in session:
        return redirect(url_for("student_login"))

    data = request.get_json()
    language = data.get("language", session.get("coding_language", "Python"))
    code = data.get("code", "")
    output = data.get("output", "")
    username = session["username"]

    looking_away_count = session.get('looking_away_count', 0)
    tab_switch_count = session.get('tab_switch_count', 0)
    camera_hidden_count = session.get('camera_hidden_count', 0)
    hand_cover_count = session.get('hand_cover_count', 0)
    cheating_count = session.get('cheating_count', 0)
    terminated = session.get('terminated', False)

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO coding_results
              (username, language, code, output, submitted_at,
               looking_away_count, tab_switch_count, camera_hidden_count,
               hand_cover_count, cheating_count, terminated)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                username, language, code, output, datetime.now(),
                looking_away_count, tab_switch_count, camera_hidden_count,
                hand_cover_count, cheating_count, terminated,
            ),
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/admin-coding-results")
def admin_coding_results():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, language, code, output, submitted_at,
               looking_away_count, tab_switch_count, camera_hidden_count,
               hand_cover_count, cheating_count, terminated
        FROM coding_results
        ORDER BY submitted_at DESC
    """)
    results = cur.fetchall()
    conn.close()

    return render_template("admin_coding_results.html", results=results)


@app.route("/delete-coding-result", methods=["POST"])
def delete_coding_result():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
        return redirect(url_for("admin_login"))
    result_id = request.form["result_id"]
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM coding_results WHERE id=%s", (result_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("admin_coding_results"))

@app.route('/upload-exam-file', methods=['POST'])
def upload_exam_file():
    file = request.files.get('student_file')
    if file:
        file.save('uploads/' + file.filename)  # Make sure 'uploads/' exists
        return redirect('/')  # Or redirect to the exam page
    return "No file uploaded", 400
# ================= RUN APP =================
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
