from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import sqlite3, os, io, requests, base64, json, time
from PIL import Image, ImageDraw
from functools import wraps
from datetime import datetime

# ── Load environment variables from .env file ─────────────────────────────────
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'vtry-secret-key-2024')

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, 'static', 'uploads')
GARMENT_FOLDER = os.path.join(BASE_DIR, 'static', 'garments')
RESULT_FOLDER  = os.path.join(BASE_DIR, 'static', 'results')
DB_PATH        = os.path.join(BASE_DIR, 'instance', 'vtry.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, GARMENT_FOLDER, RESULT_FOLDER, os.path.dirname(DB_PATH)]:
    os.makedirs(folder, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── API Keys — loaded securely from .env file ─────────────────────────────────
# Create a .env file in your project root (never commit it to git):
#
#   GEMINI_API_KEY=your_gemini_key_here
#   LIGHTX_API_KEY=your_lightx_key_here
#   FLASK_SECRET_KEY=any_random_string_here
#
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
LIGHTX_API_KEY = os.getenv('LIGHTX_API_KEY', '')


def encode_image_base64(img_path):
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_mime_type(img_path):
    ext = img_path.lower().split('.')[-1]
    return {'jpg':'image/jpeg','jpeg':'image/jpeg','png':'image/png'}.get(ext,'image/jpeg')


# ── LightX Virtual Try-On ─────────────────────────────────────────────────────
def ai_tryon_lightx(user_img_path, garment_img_path, result_path):
    """
    Use LightX Virtual Try-On API (dedicated try-on model — best quality).

    Flow:
      1. Upload user photo  → get imageUrl
      2. Upload garment     → get clothImageUrl
      3. POST /v1/virtual-tryon with both URLs → get orderId
      4. Poll /v1/order-status until done
      5. Download result image → save to result_path
    """
    if not LIGHTX_API_KEY:
        return False, "LightX API key not set (add LIGHTX_API_KEY to .env)"

    BASE    = "https://api.lightxeditor.com/external/api"
    HEADERS = {"Content-Type": "application/json", "x-api-key": LIGHTX_API_KEY}

    def upload_image(img_path, label):
        ext          = img_path.lower().split('.')[-1]
        content_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

        resp = requests.post(
            f"{BASE}/v2/uploadImageUrl",
            json={"imageType": content_type, "size": os.path.getsize(img_path)},
            headers=HEADERS, timeout=30
        )
        if resp.status_code != 200:
            raise Exception(f"Upload URL error ({label}): {resp.status_code} {resp.text[:200]}")

        body       = resp.json().get("body", resp.json())
        upload_url = body.get("uploadImageUrl") or body.get("upload_url")
        image_url  = body.get("imageUrl")       or body.get("image_url")
        if not upload_url or not image_url:
            raise Exception(f"Missing upload URL in response: {body}")

        with open(img_path, "rb") as f:
            put = requests.put(upload_url, data=f, headers={"Content-Type": content_type}, timeout=60)
        if put.status_code not in (200, 204):
            raise Exception(f"S3 PUT failed ({label}): {put.status_code}")

        return image_url

    try:
        print("LightX: uploading user photo...")
        user_url    = upload_image(user_img_path,    "user photo")
        print("LightX: uploading garment...")
        garment_url = upload_image(garment_img_path, "garment")

        print("LightX: submitting try-on job...")
        r = requests.post(
            f"{BASE}/v1/virtual-tryon",
            json={"imageUrl": user_url, "clothImageUrl": garment_url},
            headers=HEADERS, timeout=60
        )
        if r.status_code != 200:
            return False, f"Try-on submission error: {r.status_code} {r.text[:300]}"

        body     = r.json().get("body", r.json())
        order_id = body.get("orderId") or body.get("order_id")
        if not order_id:
            return False, f"No orderId in response: {body}"

        print(f"LightX: job submitted (orderId={order_id}), polling...")

        for attempt in range(30):          # up to ~90 seconds
            time.sleep(3)
            sr = requests.post(
                f"{BASE}/v1/order-status",
                json={"orderId": order_id},
                headers=HEADERS, timeout=30
            )
            if sr.status_code != 200:
                continue

            sbody  = sr.json().get("body", sr.json())
            status = sbody.get("status", "").lower()
            print(f"  poll [{attempt+1}/30]: {status}")

            if status in ("active", "completed", "success"):
                out_url = (sbody.get("output") or sbody.get("outputUrl") or
                           sbody.get("output_url") or sbody.get("resultUrl"))
                if not out_url:
                    return False, f"Job done but no output URL: {sbody}"

                dl = requests.get(out_url, timeout=60)
                if dl.status_code != 200:
                    return False, f"Failed to download result: {dl.status_code}"

                with open(result_path, "wb") as f:
                    f.write(dl.content)
                print("LightX: saved successfully.")
                return True, "LightX virtual try-on successful"

            elif status in ("failed", "error", "cancelled"):
                return False, f"Job failed: {sbody.get('message') or status}"

        return False, "Timed out waiting for LightX result (90s)"

    except Exception as e:
        return False, f"LightX error: {str(e)}"


# ── Gemini AI Try-On ──────────────────────────────────────────────────────────
def ai_tryon_gemini(user_img_path, garment_img_path, result_path):
    """Use Google Gemini 2.0 Flash for virtual try-on (fallback)."""
    if not GEMINI_API_KEY:
        return False, "Gemini API key not set (add GEMINI_API_KEY to .env)"

    try:
        user_b64     = encode_image_base64(user_img_path)
        garment_b64  = encode_image_base64(garment_img_path)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [
                {"text": (
                    "You are a virtual try-on AI. I will give you two images: "
                    "1) A person standing in their current clothes "
                    "2) A garment/clothing item. "
                    "Generate a PHOTOREALISTIC image of the SAME person wearing the garment. "
                    "Keep their face, body shape, background and pose EXACTLY the same. "
                    "Only change the clothing. Make it look completely real."
                )},
                {"inline_data": {"mime_type": get_mime_type(user_img_path),    "data": user_b64}},
                {"inline_data": {"mime_type": get_mime_type(garment_img_path), "data": garment_b64}}
            ]}],
            "generationConfig": {"response_modalities": ["IMAGE","TEXT"], "temperature": 1, "top_p": 0.95}
        }

        response = requests.post(url, json=payload, timeout=120)
        if response.status_code != 200:
            return False, f"Gemini error {response.status_code}: {response.text[:200]}"

        for candidate in response.json().get('candidates', []):
            for part in candidate.get('content', {}).get('parts', []):
                if 'inline_data' in part and part['inline_data'].get('data'):
                    with open(result_path, 'wb') as f:
                        f.write(base64.b64decode(part['inline_data']['data']))
                    return True, "Gemini AI try-on successful"

        return False, "No image in Gemini response"

    except Exception as e:
        return False, f"Gemini error: {str(e)}"


# ── Placement config ──────────────────────────────────────────────────────────
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

# ── Fit Analyzer ──────────────────────────────────────────────────────────────
def analyze_fit(user_img_path, garment_img_path, gtype, garment_size):
    import numpy as np
    try:
        user   = Image.open(user_img_path).convert('RGB')
        uw, uh = user.size
        cfg    = PLACEMENT.get(gtype, PLACEMENT['default'])
        zone   = cfg['zone']

        if zone == 'upper':
            region = user.crop((int(uw*0.15), int(uh*0.12), int(uw*0.85), int(uh*0.52)))
        elif zone == 'lower':
            region = user.crop((int(uw*0.20), int(uh*0.46), int(uw*0.80), int(uh*0.95)))
        else:
            region = user.crop((int(uw*0.15), int(uh*0.10), int(uw*0.85), int(uh*0.90)))

        arr      = np.array(region)
        rw       = region.size[0]
        col_means = arr.mean(axis=0)
        bg_color  = np.array([arr[0,0], arr[0,-1]]).mean(axis=0)
        body_cols = [i for i in range(rw) if np.sqrt(np.sum((col_means[i]-bg_color)**2)) > 25]

        body_width_ratio    = (len(body_cols) if body_cols else int(rw*0.55)) / uw
        size_mult           = {'XS':0.88,'S':0.92,'M':1.00,'L':1.08,'XL':1.15,'XXL':1.22}.get(str(garment_size).upper(),1.00)
        effective_g_ratio   = cfg['w'] * size_mult
        ratio               = effective_g_ratio / body_width_ratio if body_width_ratio > 0 else 1.0

        if   ratio < 0.88:  status,label,msg,color = 'too_tight',    '🔴 Too Tight',      'This garment will be too tight. Consider a larger size.',   '#ff4444'
        elif ratio < 0.95:  status,label,msg,color = 'slightly_tight','🟠 Slightly Tight', 'This garment may feel snug. You might want one size up.',   '#ff8c00'
        elif ratio <= 1.12: status,label,msg,color = 'perfect',       '🟢 Perfect Fit',   'Great choice! This garment should fit your body perfectly.','#00c853'
        elif ratio <= 1.22: status,label,msg,color = 'slightly_loose','🔵 Slightly Loose', 'This garment will be a little relaxed/loose on your body.','#2196f3'
        else:               status,label,msg,color = 'too_loose',     '⚪ Too Loose',      'This garment will be very loose. Consider a smaller size.', '#9e9e9e'

        return {'status':status,'label':label,'message':msg,'color':color,
                'body_width':round(body_width_ratio*100,1),
                'garment_width':round(effective_g_ratio*100,1),'ratio':round(ratio,2)}
    except Exception:
        return {'status':'unknown','label':'⚪ Fit Unknown','message':'Could not analyze fit.',
                'color':'#9e9e9e','body_width':0,'garment_width':0,'ratio':1.0}

COLOR_MAP = {
    'White':(240,240,240),'Blue':(60,110,200),'Black':(30,30,35),'Red':(200,50,50),
    'Grey':(130,130,140),'Navy':(25,40,100),'Green':(50,140,70),'Pink':(230,140,170),
    'Brown':(120,75,45),'Light Blue':(140,185,235),'Floral':(190,110,160),
}

def get_color(name):
    for k,v in COLOR_MAP.items():
        if k.lower() in name.lower(): return v
    return (150,150,160)

def get_db():
    db = sqlite3.connect(DB_PATH); db.row_factory = sqlite3.Row; return db

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
    name TEXT NOT NULL, type TEXT NOT NULL, color TEXT, size TEXT,
    image_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tryon_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL, garment_id INTEGER NOT NULL,
    user_image TEXT NOT NULL, result_image TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (garment_id) REFERENCES garments(id)
);
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL, password TEXT NOT NULL
);
"""

SAMPLES = [
    ('Blue Stripe Jeans','jeans','Blue','M','real_jeans_blue_stripe.jpg'),
    ('Black Skinny Pants','pants','Black','M','real_pants_black.png'),
    ('Wide Leg Jeans','jeans','Blue','L','real_jeans_wide.png'),
    ('Pink Bow Blouse','blouse','Pink','S','real_blouse_pink.png'),
    ('White Elegant Skirt','skirt','White','M','real_skirt_white.jpg'),
    ('Black Formal Suit','suit','Black','L','real_suit_black.jpg'),
    ('Floral Blouse','blouse','Floral','M','real_blouse_floral1.jpg'),
    ('3D Flower Blouse','blouse','Peach','M','real_blouse_floral2.jpg'),
    ('White Embroidery Blouse','blouse','White','S','real_blouse_embroidery.jpg'),
    ('White Rose Blouse','blouse','White','M','real_blouse_white_rose.jpg'),
    ('Pink Bow Sleeve Blouse','blouse','Pink','S','real_blouse_pink_bow.jpg'),
    ('Pink Ruffled Camisole','blouse','Pink','S','real_camisole_pink.jpg'),
    ('Beige Chino Pants','pants','Beige','M','real_pants_beige.jpg'),
    ('Ripped Blue Jeans','jeans','Blue','M','real_jeans_ripped.jpg'),
    ('Navy Blue Jeans','jeans','Navy','L','real_jeans_navy.jpg'),
    ('Grey Formal Pants','pants','Grey','M','real_pants_grey.jpg'),
]

def init_db():
    db = get_db(); db.executescript(SQL_SCHEMA)
    try: db.execute("INSERT INTO admins (username,password) VALUES (?,?)",('admin',generate_password_hash('admin123')))
    except: pass
    for s in SAMPLES:
        try: db.execute("INSERT OR IGNORE INTO garments (name,type,color,size,image_path) VALUES (?,?,?,?,?)",s)
        except: pass
    db.commit(); db.close()

def login_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        if 'user_id' not in session: flash('Please login first.','warning'); return redirect(url_for('login'))
        return f(*args,**kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args,**kwargs):
        if 'admin_id' not in session: return redirect(url_for('admin_login'))
        return f(*args,**kwargs)
    return decorated

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def open_image_safe(path):
    img = Image.open(path); img.load(); return img

def remove_background_smart(img):
    import numpy as np
    from PIL import ImageFilter
    img_rgb = img.convert('RGB')
    arr = np.array(img_rgb, dtype=np.float32)
    r,g,b = arr[:,:,0],arr[:,:,1],arr[:,:,2]
    sat   = np.maximum(np.maximum(r,g),b) - np.minimum(np.minimum(r,g),b)
    bri   = (r+g+b)/3.0
    mask  = ((sat>18)|(  (bri<180)&(sat>6)  )) & ~((bri>190)&(sat<22))
    mi    = Image.fromarray((mask.astype(np.uint8)*255),'L').filter(ImageFilter.GaussianBlur(2.5))
    ma    = np.array(mi); ma = np.where(ma>60,255,0).astype(np.uint8)
    mi    = Image.fromarray(ma,'L').filter(ImageFilter.GaussianBlur(1.5))
    res   = Image.new('RGBA',img_rgb.size,(0,0,0,0))
    res.paste(img_rgb.convert('RGBA'),mask=mi)
    return res

def detect_body_landmarks(user_img,w,h):
    return int(h*0.21),int(h*0.24),int(h*0.50),int(h*0.52)

def simple_overlay(u_path,g_path,r_path,gtype="default"):
    from PIL import ImageFilter,ImageEnhance
    import numpy as np
    user_pil = open_image_safe(u_path).convert("RGB")
    w,h      = user_pil.size
    face_bottom,shoulder_y,waist_y,hip_y = detect_body_landmarks(user_pil,w,h)
    garm_orig  = open_image_safe(g_path)
    garm_clean = remove_background_smart(garm_orig)
    cfg  = PLACEMENT.get(gtype,PLACEMENT["default"])
    zone = cfg.get("zone","upper")
    if zone=="upper":
        y_off=shoulder_y; g_h=waist_y-shoulder_y
        g_w=int(g_h*(garm_orig.width/garm_orig.height))
        if g_w>int(w*0.70): g_w=int(w*0.70); g_h=int(g_w*(garm_orig.height/garm_orig.width))
    elif zone=="lower":
        y_off=hip_y; g_h=int(h*0.50)
        g_w=int(g_h*(garm_orig.width/garm_orig.height))
        if g_w>int(w*0.65): g_w=int(w*0.65); g_h=int(g_w*(garm_orig.height/garm_orig.width))
    else:
        y_off=shoulder_y; g_h=int(h*0.78)
        g_w=int(g_h*(garm_orig.width/garm_orig.height))
        if g_w>int(w*0.78): g_w=int(w*0.78); g_h=int(g_w*(garm_orig.height/garm_orig.width))
    x_off = max(0,min((w-g_w)//2,w-g_w)); y_off=max(0,min(y_off,h-g_h))
    gr    = garm_clean.resize((g_w,g_h),Image.LANCZOS)
    alpha = np.array(gr.split()[3].filter(ImageFilter.GaussianBlur(2)))
    fl    = max(0,face_bottom-y_off)
    if fl>0: alpha[:fl,:]=0
    res = user_pil.copy().convert("RGBA")
    res.paste(gr,(x_off,y_off),Image.fromarray(alpha.astype(np.uint8),'L'))
    out = ImageEnhance.Sharpness(ImageEnhance.Contrast(res.convert("RGB")).enhance(1.02)).enhance(1.05)
    out.save(r_path,"JPEG",quality=97)
    return True

def apply_tryon(u_path,g_path,r_path,gtype='default'):
    """
    Priority: LightX → Gemini → Overlay fallback
    Keys are read from environment variables (never hardcoded).
    """
    for path,label in [(u_path,'user photo'),(g_path,'garment')]:
        if not os.path.exists(path): raise ValueError(f"{label} not found: {path}")
        if os.path.getsize(path)<100: raise ValueError(f"{label} file is empty")

    if LIGHTX_API_KEY:
        ok,msg = ai_tryon_lightx(u_path,g_path,r_path)
        if ok: return True,"lightx"
        print(f"LightX failed: {msg} — trying Gemini...")

    if GEMINI_API_KEY:
        ok,msg = ai_tryon_gemini(u_path,g_path,r_path)
        if ok: return True,"ai"
        print(f"Gemini failed: {msg} — using overlay fallback")

    try:
        simple_overlay(u_path,g_path,r_path,gtype)
        return True,"overlay"
    except Exception as e:
        raise ValueError(f"Cannot process images: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        u=request.form['username'].strip(); e=request.form['email'].strip(); p=request.form['password']
        if not u or not e or not p: flash('All fields required.','error'); return render_template('register.html')
        db=get_db()
        try:
            db.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",(u,e,generate_password_hash(p)))
            db.commit(); flash('Registered! Please login.','success'); return redirect(url_for('login'))
        except sqlite3.IntegrityError: flash('Username or email exists.','error')
        finally: db.close()
    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        u=request.form['username'].strip(); p=request.form['password']
        db=get_db(); user=db.execute("SELECT * FROM users WHERE username=?",(u,)).fetchone(); db.close()
        if user and check_password_hash(user['password'],p):
            session['user_id']=user['id']; session['username']=user['username']
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.','error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    db=get_db()
    results=db.execute(
        "SELECT tr.*,g.name as garment_name,g.color,g.type FROM tryon_results tr "
        "JOIN garments g ON tr.garment_id=g.id WHERE tr.user_id=? ORDER BY tr.created_at DESC LIMIT 6",
        (session['user_id'],)).fetchall()
    db.close(); return render_template('dashboard.html',results=results)

@app.route('/catalog')
@login_required
def catalog():
    from collections import defaultdict
    db=get_db(); garments=db.execute("SELECT * FROM garments ORDER BY type,id").fetchall(); db.close()
    grouped=defaultdict(list)
    for g in garments: grouped[g['type']].append(g)
    return render_template('catalog.html',grouped=dict(grouped))

@app.route('/tryon',methods=['GET','POST'])
@login_required
def tryon():
    from collections import defaultdict
    if request.method=='GET':
        gid=request.args.get('garment_id')
        db=get_db()
        garment=db.execute("SELECT * FROM garments WHERE id=?",(gid,)).fetchone() if gid else None
        garments=db.execute("SELECT * FROM garments ORDER BY type,id").fetchall(); db.close()
        grouped=defaultdict(list)
        for g in garments: grouped[g['type']].append(g)
        return render_template('tryon.html',garment=garment,grouped=dict(grouped),garments=garments)

    gid=request.form.get('garment_id'); user_image=request.files.get('user_image')
    if not gid or not user_image: flash('Please upload a photo and select a garment.','error'); return redirect(url_for('tryon'))
    if not allowed_file(user_image.filename): flash('Only JPG/PNG allowed.','error'); return redirect(url_for('tryon'))

    db=get_db(); garment=db.execute("SELECT * FROM garments WHERE id=?",(gid,)).fetchone()
    if not garment: flash('Garment not found.','error'); db.close(); return redirect(url_for('catalog'))

    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    u_fn=f"user_{session['user_id']}_{ts}.jpg"; r_fn=f"result_{session['user_id']}_{ts}.jpg"
    u_path=os.path.join(UPLOAD_FOLDER,u_fn); g_path=os.path.join(GARMENT_FOLDER,garment['image_path']); r_path=os.path.join(RESULT_FOLDER,r_fn)

    user_image.stream.seek(0); file_bytes=user_image.stream.read()
    if len(file_bytes)<100: flash('Uploaded file is empty.','error'); return redirect(url_for('tryon'))
    with open(u_path,'wb') as out: out.write(file_bytes)
    if os.path.getsize(u_path)<100: flash('File saved incorrectly. Please try again.','error'); return redirect(url_for('tryon'))

    header=open(u_path,'rb').read(16).hex()
    is_heic='ftyp' in bytes.fromhex(header[:32]).decode('latin-1','ignore')
    try:
        if is_heic:
            try:
                import pillow_heif; pillow_heif.register_heif_opener()
                img=Image.open(u_path); img.load()
            except ImportError:
                os.rename(u_path,u_path.replace('.jpg','.heic'))
                flash('iPhone HEIC detected. Please convert to JPG first.','error'); return redirect(url_for('tryon'))
        else:
            img=Image.open(u_path); img.load()
        img.convert('RGB').save(u_path,'JPEG',quality=95)
    except Exception:
        flash('Cannot read this image. Please convert to JPG first.','error'); return redirect(url_for('tryon'))

    try:
        success,mode = apply_tryon(u_path,g_path,r_path,garment['type'])
        fit = analyze_fit(u_path,g_path,garment['type'],garment['size'])
        db.execute("INSERT INTO tryon_results (user_id,garment_id,user_image,result_image) VALUES (?,?,?,?)",
                   (session['user_id'],gid,u_fn,r_fn))
        db.commit()
        msgs = {
            "lightx":  '✨ LightX AI try-on generated — photorealistic virtual fitting!',
            "ai":      '✨ Gemini AI try-on generated — garment wrapped on your body!',
            "overlay": 'Try-on generated! Add LIGHTX_API_KEY or GEMINI_API_KEY to your .env for AI results.',
        }
        flash(msgs.get(mode,'Try-on complete!'),'success')
        import urllib.parse
        return redirect(url_for('result',filename=r_fn,fit=urllib.parse.quote(json.dumps(fit))))
    except Exception as e: flash(f'Error: {str(e)}','error'); return redirect(url_for('tryon'))
    finally: db.close()

@app.route('/result/<filename>')
@login_required
def result(filename):
    import urllib.parse
    db=get_db()
    row=db.execute(
        "SELECT tr.*,g.name as garment_name,g.color,g.type,g.size FROM tryon_results tr "
        "JOIN garments g ON tr.garment_id=g.id WHERE tr.result_image=? AND tr.user_id=?",
        (filename,session['user_id'])).fetchone()
    db.close()
    fit={}
    fe=request.args.get('fit','')
    if fe:
        try: fit=json.loads(urllib.parse.unquote(fe))
        except: pass
    if not fit and row:
        u_path=os.path.join(BASE_DIR,'static','uploads',row['user_image'])
        g_path=os.path.join(BASE_DIR,'static','garments',row['image_path'] if 'image_path' in row.keys() else '')
        if os.path.exists(u_path):
            fit=analyze_fit(u_path,g_path,row['type'],row['size'] if 'size' in row.keys() else 'M')
    return render_template('result.html',result=row,filename=filename,fit=fit)

@app.route('/my-results')
@login_required
def my_results():
    db=get_db()
    results=db.execute(
        "SELECT tr.*,g.name as garment_name,g.color,g.type FROM tryon_results tr "
        "JOIN garments g ON tr.garment_id=g.id WHERE tr.user_id=? ORDER BY tr.created_at DESC",
        (session['user_id'],)).fetchall()
    db.close(); return render_template('my_results.html',results=results)

@app.route('/static/results/<filename>')
def serve_result(filename): return send_from_directory(RESULT_FOLDER,filename)

@app.route('/admin/login',methods=['GET','POST'])
def admin_login():
    if request.method=='POST':
        u=request.form['username']; p=request.form['password']
        db=get_db(); admin=db.execute("SELECT * FROM admins WHERE username=?",(u,)).fetchone(); db.close()
        if admin and check_password_hash(admin['password'],p):
            session['admin_id']=admin['id']; session['admin_name']=admin['username']
            return redirect(url_for('admin_dashboard'))
        flash('Invalid credentials.','error')
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id',None); session.pop('admin_name',None); return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    db=get_db()
    uc=db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    gc=db.execute("SELECT COUNT(*) FROM garments").fetchone()[0]
    rc=db.execute("SELECT COUNT(*) FROM tryon_results").fetchone()[0]
    ru=db.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT 5").fetchall()
    rr=db.execute(
        "SELECT tr.*,u.username,g.name as garment_name FROM tryon_results tr "
        "JOIN users u ON tr.user_id=u.id JOIN garments g ON tr.garment_id=g.id "
        "ORDER BY tr.created_at DESC LIMIT 5").fetchall()
    db.close()
    return render_template('admin_dashboard.html',users_count=uc,garments_count=gc,results_count=rc,
                           recent_users=ru,recent_results=rr)

@app.route('/admin/garments',methods=['GET','POST'])
@admin_required
def admin_garments():
    db=get_db()
    if request.method=='POST':
        name=request.form['name']; gtype=request.form['type']
        color=request.form['color']; size=request.form['size']; image=request.files.get('image')
        if image and allowed_file(image.filename):
            fn=secure_filename(image.filename); image.save(os.path.join(GARMENT_FOLDER,fn))
            db.execute("INSERT INTO garments (name,type,color,size,image_path) VALUES (?,?,?,?,?)",(name,gtype,color,size,fn))
            db.commit(); flash('Garment added!','success')
        else: flash('Invalid image.','error')
    garments=db.execute("SELECT * FROM garments ORDER BY type,id DESC").fetchall(); db.close()
    return render_template('admin_garments.html',garments=garments)

@app.route('/admin/garments/delete/<int:gid>')
@admin_required
def delete_garment(gid):
    db=get_db(); db.execute("DELETE FROM garments WHERE id=?",(gid,)); db.commit(); db.close()
    flash('Deleted.','success'); return redirect(url_for('admin_garments'))

@app.route('/admin/users')
@admin_required
def admin_users():
    db=get_db()
    users=db.execute(
        "SELECT u.*,COUNT(tr.id) as tryons FROM users u "
        "LEFT JOIN tryon_results tr ON u.id=tr.user_id GROUP BY u.id ORDER BY u.created_at DESC").fetchall()
    db.close(); return render_template('admin_users.html',users=users)

if __name__=='__main__':
    init_db(); app.run(debug=True,port=5000)
