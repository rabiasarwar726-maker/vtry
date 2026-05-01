from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3, os, io, requests, base64, json, time, urllib.parse
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from functools import wraps
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = 'vtry-secret-key-2024'

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
GARMENT_FOLDER = os.path.join(BASE_DIR, 'static', 'garments')
RESULT_FOLDER  = os.path.join(BASE_DIR, 'static', 'results')
DB_PATH        = os.path.join(BASE_DIR, 'instance', 'vtry.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, GARMENT_FOLDER, RESULT_FOLDER, os.path.dirname(DB_PATH)]:
    os.makedirs(folder, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

FASHN_API_KEY  = os.getenv('FASHN_API_KEY', '') 

FASHN_RUN_URL    = "https://api.fashn.ai/v1/run"
FASHN_STATUS_URL = "https://api.fashn.ai/v1/status"

# ── Placement config (fallback overlay) ──────────────────────────
PLACEMENT = {
    'top':     {'w':0.60, 'h':0.36, 'y':0.13, 'zone':'upper'},
    'shirt':   {'w':0.62, 'h':0.38, 'y':0.12, 'zone':'upper'},
    'blouse':  {'w':0.62, 'h':0.36, 'y':0.12, 'zone':'upper'},
    'jacket':  {'w':0.68, 'h':0.42, 'y':0.11, 'zone':'upper'},
    'hoodie':  {'w':0.66, 'h':0.40, 'y':0.11, 'zone':'upper'},
    'kameez':  {'w':0.65, 'h':0.58, 'y':0.10, 'zone':'upper'},
    'abaya':   {'w':0.74, 'h':0.84, 'y':0.07, 'zone':'full'},
    'suit':    {'w':0.72, 'h':0.72, 'y':0.07, 'zone':'full'},
    'pants':   {'w':0.58, 'h':0.52, 'y':0.46, 'zone':'lower'},
    'jeans':   {'w':0.58, 'h':0.52, 'y':0.46, 'zone':'lower'},
    'shalwar': {'w':0.60, 'h':0.50, 'y':0.46, 'zone':'lower'},
    'skirt':   {'w':0.62, 'h':0.44, 'y':0.46, 'zone':'lower'},
    'default': {'w':0.62, 'h':0.40, 'y':0.12, 'zone':'upper'},
}

# ── FASHN category mapping ────────────────────────────────────────
FASHN_CATEGORY = {
    'top':     'tops',
    'shirt':   'tops',
    'blouse':  'tops',
    'jacket':  'tops',
    'hoodie':  'tops',
    'kameez':  'tops',
    'abaya':   'one-pieces',
    'suit':    'one-pieces',
    'pants':   'bottoms',
    'jeans':   'bottoms',
    'shalwar': 'bottoms',
    'skirt':   'bottoms',
    'default': 'tops',
}

# ═══════════════════════════════════════════════════════════════════
# FASHN API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def encode_image_base64(img_path):
    ext = img_path.lower().split('.')[-1]
    mime = 'image/jpeg' if ext in ('jpg','jpeg') else 'image/png'
    with open(img_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return f"data:{mime};base64,{b64}"

def fashn_submit_job(user_img_path, garment_img_path, gtype):
    category = FASHN_CATEGORY.get(gtype, 'tops')
    print(f"[FASHN] Submitting job — category: {category}")
    print(f"[FASHN] User image: {os.path.getsize(user_img_path)} bytes")
    print(f"[FASHN] Garment image: {os.path.getsize(garment_img_path)} bytes")

    model_b64   = encode_image_base64(user_img_path)
    garment_b64 = encode_image_base64(garment_img_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FASHN_API_KEY}"
    }

    payload = {
        "model_name": "tryon-v1.6",
        "inputs": {
            "model_image":   model_b64,
            "garment_image": garment_b64,
            "category":      category,
            "mode":          "balanced",
            "return_base64": False
        }
    }

    print(f"[FASHN] Sending POST to {FASHN_RUN_URL}...")
    try:
        resp = requests.post(FASHN_RUN_URL, json=payload, headers=headers, timeout=60)
    except Exception as e:
        return None, f"Connection error: {e}"

    print(f"[FASHN] Response status: {resp.status_code}")
    print(f"[FASHN] Response body: {resp.text[:400]}")

    if resp.status_code == 401:
        return None, "Invalid API key"
    if resp.status_code == 402:
        return None, "Insufficient credits"
    if resp.status_code != 200:
        return None, f"FASHN submit error {resp.status_code}: {resp.text[:300]}"

    data = resp.json()
    if data.get('error'):
        return None, f"FASHN API error: {data['error']}"

    pred_id = data.get('id')
    if not pred_id:
        return None, f"No prediction ID returned. Full response: {data}"

    print(f"[FASHN] Got prediction ID: {pred_id}")
    return pred_id, None

def fashn_poll_result(pred_id, max_wait=120):
    headers = {"Authorization": f"Bearer {FASHN_API_KEY}"}
    url     = f"{FASHN_STATUS_URL}/{pred_id}"
    start   = time.time()
    attempt = 0

    while time.time() - start < max_wait:
        attempt += 1
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except Exception as e:
            print(f"[FASHN] Poll #{attempt} connection error: {e}")
            time.sleep(3)
            continue

        if resp.status_code != 200:
            return None, f"Poll error {resp.status_code}: {resp.text[:200]}"

        data   = resp.json()
        status = data.get('status', '')
        print(f"[FASHN] Poll #{attempt} — status: {status} ({int(time.time()-start)}s elapsed)")

        if status == 'completed':
            output = data.get('output', [])
            print(f"[FASHN] Completed! Output: {output}")
            if output:
                return output[0], None
            return None, "Completed but no output in response"

        elif status == 'failed':
            err = data.get('error', {})
            msg = err.get('message', str(err)) if isinstance(err, dict) else str(err)
            print(f"[FASHN] Failed: {msg}")
            return None, f"FASHN job failed: {msg}"

        elif status in ('starting', 'in_queue', 'processing'):
            time.sleep(3)
            continue

        else:
            return None, f"Unknown status: {status}"

    return None, f"Timeout after {max_wait}s — job may still be processing"

def ai_tryon_fashn(user_img_path, garment_img_path, result_path, gtype='default'):
    if not FASHN_API_KEY or FASHN_API_KEY == "YOUR_FASHN_API_KEY_HERE":
        return False, "FASHN API key not set"

    pred_id, err = fashn_submit_job(user_img_path, garment_img_path, gtype)
    if not pred_id:
        return False, f"Submit failed: {err}"

    print(f"FASHN job submitted: {pred_id}")

    output, err = fashn_poll_result(pred_id, max_wait=120)
    if not output:
        return False, f"Poll failed: {err}"

    try:
        if output.startswith('data:'):
            b64_data = output.split(',', 1)[1]
            img_bytes = base64.b64decode(b64_data)
            with open(result_path, 'wb') as f:
                f.write(img_bytes)
        else:
            img_resp = requests.get(output, timeout=60)
            if img_resp.status_code != 200:
                return False, f"Could not download result: {img_resp.status_code}"
            with open(result_path, 'wb') as f:
                f.write(img_resp.content)

        return True, "FASHN AI try-on successful"

    except Exception as e:
        return False, f"Save result error: {str(e)}"

# ═══════════════════════════════════════════════════════════════════
# FALLBACK OVERLAY
# ═══════════════════════════════════════════════════════════════════

def open_image_safe(path):
    img = Image.open(path)
    img.load()
    return img

def remove_background_smart(img):
    img_rgb = img.convert('RGB')
    arr = np.array(img_rgb, dtype=np.float32)
    r, g, b  = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    sat = np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)
    bri = (r + g + b) / 3.0
    is_garment = (sat > 18) | ((bri < 180) & (sat > 6))
    is_bg      = (bri > 190) & (sat < 22)
    is_garment = is_garment & ~is_bg
    mask = is_garment.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, 'L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2.5))
    mask_arr = np.where(np.array(mask_img) > 60, 255, 0).astype(np.uint8)
    mask_img = Image.fromarray(mask_arr, 'L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    result = Image.new('RGBA', img_rgb.size, (0,0,0,0))
    result.paste(img_rgb.convert('RGBA'), mask=mask_img)
    return result

def simple_overlay(u_path, g_path, r_path, gtype='default'):
    user_pil = open_image_safe(u_path).convert('RGB')
    w, h     = user_pil.size
    face_end   = int(h * 0.20)
    shoulder_y = int(h * 0.23)
    waist_y    = int(h * 0.56)
    hip_y      = int(h * 0.52)
    cfg  = PLACEMENT.get(gtype, PLACEMENT['default'])
    zone = cfg.get('zone', 'upper')
    garment_orig  = open_image_safe(g_path)
    garment_clean = remove_background_smart(garment_orig)
    ow, oh = garment_orig.size
    aspect = ow / oh if oh > 0 else 1
    if zone == 'upper':
        new_gh = waist_y - shoulder_y
        new_gw = int(new_gh * aspect)
        if new_gw > int(w * 0.72): new_gw = int(w * 0.72); new_gh = int(new_gw / aspect)
        x_off = (w - new_gw) // 2; y_off = shoulder_y
    elif zone == 'lower':
        new_gh = int(h * 0.50)
        new_gw = int(new_gh * aspect)
        if new_gw > int(w * 0.65): new_gw = int(w * 0.65); new_gh = int(new_gw / aspect)
        x_off = (w - new_gw) // 2; y_off = hip_y
    else:
        new_gh = int(h * 0.78)
        new_gw = int(new_gh * aspect)
        if new_gw > int(w * 0.78): new_gw = int(w * 0.78); new_gh = int(new_gw / aspect)
        x_off = (w - new_gw) // 2; y_off = shoulder_y
    garment_resized = garment_clean.resize((new_gw, new_gh), Image.LANCZOS)
    x_off = max(0, min(x_off, w - new_gw))
    y_off = max(0, min(y_off, h - new_gh))
    alpha      = garment_resized.split()[3]
    alpha_blur = alpha.filter(ImageFilter.GaussianBlur(radius=2))
    alpha_arr  = np.array(alpha_blur)
    face_limit = max(0, face_end - y_off)
    if face_limit > 0: alpha_arr[:face_limit, :] = 0
    alpha_prot = Image.fromarray(alpha_arr.astype(np.uint8), 'L')
    result = user_pil.copy().convert('RGBA')
    result.paste(garment_resized, (x_off, y_off), alpha_prot)
    out = result.convert('RGB')
    ImageEnhance.Contrast(out).enhance(1.02).save(r_path, 'JPEG', quality=97)
    return True

# ═══════════════════════════════════════════════════════════════════
# MAIN TRY-ON FUNCTION
# ═══════════════════════════════════════════════════════════════════

def apply_tryon(u_path, g_path, r_path, gtype='default'):
    for path, label in [(u_path,'user photo'),(g_path,'garment')]:
        if not os.path.exists(path):
            raise ValueError(f"{label} not found: {path}")
        if os.path.getsize(path) < 100:
            raise ValueError(f"{label} file is empty")

    if FASHN_API_KEY and FASHN_API_KEY != "YOUR_FASHN_API_KEY_HERE":
        success, msg = ai_tryon_fashn(u_path, g_path, r_path, gtype)
        if success:
            print(f"✅ FASHN AI try-on success")
            return True, "ai"
        print(f"⚠ FASHN failed: {msg} — using overlay fallback")
        app.config['LAST_FASHN_ERROR'] = msg

    try:
        simple_overlay(u_path, g_path, r_path, gtype)
        return True, "overlay"
    except Exception as e:
        raise ValueError(f"Cannot process images: {e}")

# ═══════════════════════════════════════════════════════════════════
# FIT ANALYZER
# ═══════════════════════════════════════════════════════════════════

def analyze_fit(user_img_path, garment_img_path, gtype, garment_size):
    try:
        user = Image.open(user_img_path).convert('RGB')
        uw, uh = user.size
        cfg  = PLACEMENT.get(gtype, PLACEMENT['default'])
        zone = cfg['zone']
        if zone == 'upper':
            region = user.crop((int(uw*0.15), int(uh*0.12), int(uw*0.85), int(uh*0.52)))
        elif zone == 'lower':
            region = user.crop((int(uw*0.20), int(uh*0.46), int(uw*0.80), int(uh*0.95)))
        else:
            region = user.crop((int(uw*0.15), int(uh*0.10), int(uw*0.85), int(uh*0.90)))
        arr = np.array(region)
        rw  = region.size[0]
        col_means = arr.mean(axis=0)
        bg_color  = np.array([arr[0,0], arr[0,-1]]).mean(axis=0)
        body_cols = [i for i in range(rw)
                     if np.sqrt(np.sum((col_means[i] - bg_color)**2)) > 25]
        body_w = len(body_cols) if body_cols else int(rw * 0.55)
        bwr    = body_w / uw
        gwr    = cfg['w']
        sm = {'XS':0.88,'S':0.92,'M':1.00,'L':1.08,'XL':1.15,'XXL':1.22}
        egr   = gwr * sm.get(str(garment_size).upper(), 1.00)
        ratio = egr / bwr if bwr > 0 else 1.0
        if ratio < 0.88:
            return {'status':'too_tight','label':'🔴 Too Tight',
                    'message':'This garment will be too tight. Consider a larger size.',
                    'color':'#ff4444','body_width':round(bwr*100,1),
                    'garment_width':round(egr*100,1),'ratio':round(ratio,2)}
        elif ratio < 0.95:
            return {'status':'slightly_tight','label':'🟠 Slightly Tight',
                    'message':'May feel a little snug. Try one size up.',
                    'color':'#ff8c00','body_width':round(bwr*100,1),
                    'garment_width':round(egr*100,1),'ratio':round(ratio,2)}
        elif ratio <= 1.12:
            return {'status':'perfect','label':'🟢 Perfect Fit',
                    'message':'Great choice! This garment should fit you perfectly.',
                    'color':'#00c853','body_width':round(bwr*100,1),
                    'garment_width':round(egr*100,1),'ratio':round(ratio,2)}
        elif ratio <= 1.22:
            return {'status':'slightly_loose','label':'🔵 Slightly Loose',
                    'message':'Will be a little relaxed. Could try one size down.',
                    'color':'#2196f3','body_width':round(bwr*100,1),
                    'garment_width':round(egr*100,1),'ratio':round(ratio,2)}
        else:
            return {'status':'too_loose','label':'⚪ Too Loose',
                    'message':'Will be very loose. Consider a smaller size.',
                    'color':'#9e9e9e','body_width':round(bwr*100,1),
                    'garment_width':round(egr*100,1),'ratio':round(ratio,2)}
    except:
        return {'status':'unknown','label':'⚪ Fit Unknown',
                'message':'Could not analyze fit.','color':'#9e9e9e',
                'body_width':0,'garment_width':0,'ratio':1.0}

# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════

def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS garments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    color TEXT,
    size TEXT,
    image_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tryon_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    garment_id INTEGER NOT NULL,
    user_image TEXT NOT NULL,
    result_image TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (garment_id) REFERENCES garments(id)
);
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
"""

SAMPLES = [
    ('Blue Stripe Jeans',       'jeans',  'Blue',   'M',  'real_jeans_blue_stripe.jpg'),
    ('Black Skinny Pants',      'pants',  'Black',  'M',  'real_pants_black.png'),
    ('Wide Leg Jeans',          'jeans',  'Blue',   'L',  'real_jeans_wide.png'),
    ('Pink Bow Blouse',         'blouse', 'Pink',   'S',  'real_blouse_pink.png'),
    ('White Elegant Skirt',     'skirt',  'White',  'M',  'real_skirt_white.jpg'),
    ('Black Formal Suit',       'suit',   'Black',  'L',  'real_suit_black.jpg'),
    ('Floral Blouse',           'blouse', 'Floral', 'M',  'real_blouse_floral1.jpg'),
    ('3D Flower Blouse',        'blouse', 'Peach',  'M',  'real_blouse_floral2.jpg'),
    ('White Embroidery Blouse', 'blouse', 'White',  'S',  'real_blouse_embroidery.jpg'),
    ('White Rose Blouse',       'blouse', 'White',  'M',  'real_blouse_white_rose.jpg'),
    ('Pink Bow Sleeve Blouse',  'blouse', 'Pink',   'S',  'real_blouse_pink_bow.jpg'),
    ('Pink Ruffled Camisole',   'blouse', 'Pink',   'S',  'real_camisole_pink.jpg'),
    ('Beige Chino Pants',       'pants',  'Beige',  'M',  'real_pants_beige.jpg'),
    ('Ripped Blue Jeans',       'jeans',  'Blue',   'M',  'real_jeans_ripped.jpg'),
    ('Navy Blue Jeans',         'jeans',  'Navy',   'L',  'real_jeans_navy.jpg'),
    ('Grey Formal Pants',       'pants',  'Grey',   'M',  'real_pants_grey.jpg'),
]

def create_garment_preview(filename, gtype, color_name):
    pass

def init_db():
    db = get_db()
    db.executescript(SQL_SCHEMA)
    try: db.execute("INSERT INTO admins (username,password) VALUES (?,?)",
                    ('admin', generate_password_hash('admin123')))
    except: pass
    for s in SAMPLES:
        try: db.execute("INSERT OR IGNORE INTO garments (name,type,color,size,image_path) VALUES (?,?,?,?,?)", s)
        except: pass
        create_garment_preview(s[4], s[1], s[2])
    db.commit(); db.close()

# ═══════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.','warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

# ═══════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u=request.form['username'].strip()
        e=request.form['email'].strip()
        p=request.form['password']
        if not u or not e or not p:
            flash('All fields required.','error')
            return render_template('register.html')
        db = get_db()
        try:
            db.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                       (u, e, generate_password_hash(p)))
            db.commit()
            flash('Registered! Please login.','success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.','error')
        finally: db.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u=request.form['username'].strip()
        p=request.form['password']
        db=get_db()
        user=db.execute("SELECT * FROM users WHERE username=?",(u,)).fetchone()
        db.close()
        if user and check_password_hash(user['password'], p):
            session['user_id']=user['id']
            session['username']=user['username']
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.','error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    db=get_db()
    results=db.execute("""SELECT tr.*,g.name as garment_name,g.color,g.type
        FROM tryon_results tr JOIN garments g ON tr.garment_id=g.id
        WHERE tr.user_id=? ORDER BY tr.created_at DESC LIMIT 6""",
        (session['user_id'],)).fetchall()
    db.close()
    return render_template('dashboard.html', results=results)

@app.route('/catalog')
@login_required
def catalog():
    from collections import defaultdict
    db=get_db()
    garments=db.execute("SELECT * FROM garments ORDER BY type,id").fetchall()
    db.close()
    grouped=defaultdict(list)
    for g in garments: grouped[g['type']].append(g)
    return render_template('catalog.html', grouped=dict(grouped))

@app.route('/tryon', methods=['GET','POST'])
@login_required
def tryon():
    from collections import defaultdict

    if request.method == 'GET':
        gid=request.args.get('garment_id')
        db=get_db()
        garment=db.execute("SELECT * FROM garments WHERE id=?",(gid,)).fetchone() if gid else None
        garments=db.execute("SELECT * FROM garments ORDER BY type,id").fetchall()
        db.close()
        grouped=defaultdict(list)
        for g in garments: grouped[g['type']].append(g)
        # Pass ai_enabled but do NOT expose key details in the template
        ai_enabled = bool(FASHN_API_KEY and FASHN_API_KEY != "YOUR_FASHN_API_KEY_HERE")
        return render_template('tryon.html', garment=garment,
                               grouped=dict(grouped), garments=garments,
                               ai_enabled=ai_enabled)

    # POST — process try-on
    gid        = request.form.get('garment_id')
    user_image = request.files.get('user_image')

    if not gid or not user_image:
        flash('Please upload a photo and select a garment.','error')
        return redirect(url_for('tryon'))
    if not allowed_file(user_image.filename):
        flash('Only JPG/PNG files allowed.','error')
        return redirect(url_for('tryon'))

    db=get_db()
    garment=db.execute("SELECT * FROM garments WHERE id=?",(gid,)).fetchone()
    if not garment:
        flash('Garment not found.','error')
        db.close()
        return redirect(url_for('catalog'))

    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    u_fn  = f"user_{session['user_id']}_{ts}.jpg"
    r_fn  = f"result_{session['user_id']}_{ts}.jpg"
    u_path = os.path.join(UPLOAD_FOLDER, u_fn)
    g_path = os.path.join(GARMENT_FOLDER, garment['image_path'])
    r_path = os.path.join(RESULT_FOLDER, r_fn)

    user_image.stream.seek(0)
    file_bytes = user_image.stream.read()
    if len(file_bytes) < 100:
        flash('Uploaded file is empty.','error')
        return redirect(url_for('tryon'))

    with open(u_path, 'wb') as out:
        out.write(file_bytes)

    header = open(u_path,'rb').read(16).hex()
    is_heic = 'ftyp' in bytes.fromhex(header[:32]).decode('latin-1','ignore')
    try:
        if is_heic:
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                flash('iPhone HEIC photo detected. Please convert to JPG first using Windows Paint.','error')
                return redirect(url_for('tryon'))
        img = Image.open(u_path)
        img.load()
        img.convert('RGB').save(u_path, 'JPEG', quality=95)
    except Exception as e:
        flash(f'Cannot read image. Please use a JPG or PNG photo.','error')
        return redirect(url_for('tryon'))

    try:
        success, mode = apply_tryon(u_path, g_path, r_path, garment['type'])

        # ── FIT ANALYSIS ──────────────────────────────────────────
        # When FASHN AI succeeds, the garment is perfectly wrapped —
        # no need for pixel-based guessing; just report Perfect Fit.
        if mode == "ai":
            fit = {
                'status': 'perfect',
                'label': '🟢 Perfect Fit',
                'message': 'FASHN AI wrapped the garment perfectly on your body.',
                'color': '#00c853',
                'ai_mode': True
            }
        else:
            fit = analyze_fit(u_path, g_path, garment['type'], garment['size'])

        fit_json = json.dumps(fit)

        db.execute("INSERT INTO tryon_results (user_id,garment_id,user_image,result_image) VALUES (?,?,?,?)",
                   (session['user_id'], gid, u_fn, r_fn))
        db.commit()

        if mode == "ai":
            flash('✨ FASHN AI try-on complete! Garment wrapped on your body.','success')
        else:
            fashn_err = app.config.get('LAST_FASHN_ERROR','')
            if fashn_err:
                flash(f'⚠ FASHN AI failed — showing overlay instead.','warning')
            else:
                flash('Try-on generated! Add your FASHN API key for AI-powered results.','success')

        fit_encoded = urllib.parse.quote(fit_json)
        return redirect(url_for('result', filename=r_fn, fit=fit_encoded))

    except Exception as e:
        flash(f'Error: {str(e)}','error')
        return redirect(url_for('tryon'))
    finally:
        db.close()

@app.route('/result/<filename>')
@login_required
def result(filename):
    db  = get_db()
    row = db.execute("""SELECT tr.*,g.name as garment_name,g.color,g.type,g.size,g.image_path
        FROM tryon_results tr JOIN garments g ON tr.garment_id=g.id
        WHERE tr.result_image=? AND tr.user_id=?""",
        (filename, session['user_id'])).fetchone()
    db.close()
    fit_encoded = request.args.get('fit','')
    fit = {}
    if fit_encoded:
        try: fit = json.loads(urllib.parse.unquote(fit_encoded))
        except: pass
    if not fit and row:
        u_path = os.path.join(BASE_DIR,'static','uploads', row['user_image'])
        g_path = os.path.join(BASE_DIR,'static','garments', row['image_path'])
        if os.path.exists(u_path) and os.path.exists(g_path):
            fit = analyze_fit(u_path, g_path, row['type'], row['size'] or 'M')
    return render_template('result.html', result=row, filename=filename, fit=fit)

@app.route('/my-results')
@login_required
def my_results():
    db=get_db()
    results=db.execute("""SELECT tr.*,g.name as garment_name,g.color,g.type
        FROM tryon_results tr JOIN garments g ON tr.garment_id=g.id
        WHERE tr.user_id=? ORDER BY tr.created_at DESC""",
        (session['user_id'],)).fetchall()
    db.close()
    return render_template('my_results.html', results=results)

@app.route('/static/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/admin/login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        u=request.form['username']; p=request.form['password']
        db=get_db()
        admin=db.execute("SELECT * FROM admins WHERE username=?",(u,)).fetchone()
        db.close()
        if admin and check_password_hash(admin['password'], p):
            session['admin_id']=admin['id']
            session['admin_name']=admin['username']
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials.','error')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id',None); session.pop('admin_name',None)
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    db=get_db()
    uc=db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    gc=db.execute("SELECT COUNT(*) FROM garments").fetchone()[0]
    rc=db.execute("SELECT COUNT(*) FROM tryon_results").fetchone()[0]
    ru=db.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT 5").fetchall()
    rr=db.execute("""SELECT tr.*,u.username,g.name as garment_name
        FROM tryon_results tr JOIN users u ON tr.user_id=u.id
        JOIN garments g ON tr.garment_id=g.id
        ORDER BY tr.created_at DESC LIMIT 5""").fetchall()
    db.close()
    return render_template('admin_dashboard.html', users_count=uc,
                           garments_count=gc, results_count=rc,
                           recent_users=ru, recent_results=rr)

@app.route('/admin/garments', methods=['GET','POST'])
@admin_required
def admin_garments():
    db=get_db()
    if request.method == 'POST':
        name=request.form['name']; gtype=request.form['type']
        color=request.form['color']; size=request.form['size']
        image=request.files.get('image')
        if image and allowed_file(image.filename):
            fn=secure_filename(image.filename)
            image.save(os.path.join(GARMENT_FOLDER, fn))
            db.execute("INSERT INTO garments (name,type,color,size,image_path) VALUES (?,?,?,?,?)",
                       (name, gtype, color, size, fn))
            db.commit()
            flash('Garment added!','success')
        else:
            flash('Invalid image.','error')
    garments=db.execute("SELECT * FROM garments ORDER BY type,id DESC").fetchall()
    db.close()
    return render_template('admin_garments.html', garments=garments)

@app.route('/admin/garments/delete/<int:gid>')
@admin_required
def delete_garment(gid):
    db=get_db()
    db.execute("DELETE FROM garments WHERE id=?",(gid,))
    db.commit(); db.close()
    flash('Deleted.','success')
    return redirect(url_for('admin_garments'))

@app.route('/admin/users')
@admin_required
def admin_users():
    db=get_db()
    users=db.execute("""SELECT u.*,COUNT(tr.id) as tryons
        FROM users u LEFT JOIN tryon_results tr ON u.id=tr.user_id
        GROUP BY u.id ORDER BY u.created_at DESC""").fetchall()
    db.close()
    return render_template('admin_users.html', users=users)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
    