from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3, os, io, requests, base64, json, time, urllib.parse
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from functools import wraps
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
load_dotenv()

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
    'hoodie':  {'w':0.66, 'h':0.42, 'y':0.11, 'zone':'upper'},
    'kameez':  {'w':0.65, 'h':0.58, 'y':0.10, 'zone':'upper'},
    'frock':   {'w':0.72, 'h':0.82, 'y':0.08, 'zone':'full'},
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
    'frock':   'one-pieces',
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
# PROFESSIONAL FIT ADVISOR — Claude AI Vision + Smart Pixel Fallback
# ═══════════════════════════════════════════════════════════════════

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# International standard size chart (chest/bust in inches)
SIZE_MEASUREMENTS = {
    'XS':  {'chest': (30, 32), 'waist': (24, 26), 'hips': (34, 36), 'label': 'Extra Small'},
    'S':   {'chest': (33, 35), 'waist': (27, 29), 'hips': (37, 39), 'label': 'Small'},
    'M':   {'chest': (36, 38), 'waist': (30, 32), 'hips': (40, 42), 'label': 'Medium'},
    'L':   {'chest': (39, 41), 'waist': (33, 35), 'hips': (43, 45), 'label': 'Large'},
    'XL':  {'chest': (42, 44), 'waist': (36, 38), 'hips': (46, 48), 'label': 'Extra Large'},
    'XXL': {'chest': (45, 47), 'waist': (39, 41), 'hips': (49, 51), 'label': '2X Large'},
}

SIZE_ORDER = ['XS', 'S', 'M', 'L', 'XL', 'XXL']

# Garment-type specific fit guidance
GARMENT_FIT_GUIDE = {
    'hoodie':  {'preferred': 'relaxed', 'tight_warn': 'restricts arm movement', 'loose_ok': True,  'key_area': 'shoulders and chest'},
    'shirt':   {'preferred': 'tailored', 'tight_warn': 'pulls at buttons',      'loose_ok': False, 'key_area': 'collar, chest, and shoulders'},
    'blouse':  {'preferred': 'relaxed',  'tight_warn': 'strains at the bust',   'loose_ok': True,  'key_area': 'bust and shoulders'},
    'jacket':  {'preferred': 'tailored', 'tight_warn': 'restricts shoulder movement', 'loose_ok': False, 'key_area': 'shoulders — most critical'},
    'kameez':  {'preferred': 'relaxed',  'tight_warn': 'too tight across chest', 'loose_ok': True, 'key_area': 'chest and hip allowance'},
    'suit':    {'preferred': 'tailored', 'tight_warn': 'restricts movement',    'loose_ok': False, 'key_area': 'jacket shoulders and trouser waist'},
    'frock':   {'preferred': 'relaxed',  'tight_warn': 'tight at hips/bust',    'loose_ok': True,  'key_area': 'bust, waist, and hip'},
    'jeans':   {'preferred': 'fitted',   'tight_warn': 'uncomfortable at waist','loose_ok': False, 'key_area': 'waist and hips'},
    'pants':   {'preferred': 'tailored', 'tight_warn': 'tight in thigh/waist',  'loose_ok': False, 'key_area': 'waist, seat, and thigh'},
    'skirt':   {'preferred': 'relaxed',  'tight_warn': 'tight across the hips', 'loose_ok': True,  'key_area': 'waist and hip'},
    'top':     {'preferred': 'relaxed',  'tight_warn': 'strains across bust',   'loose_ok': True,  'key_area': 'bust and shoulder'},
    'default': {'preferred': 'regular',  'tight_warn': 'uncomfortable fit',     'loose_ok': True,  'key_area': 'overall silhouette'},
}

def get_size_suggestion(current_size, direction):
    """Get next size up or down safely."""
    s = str(current_size).upper().strip()
    idx = SIZE_ORDER.index(s) if s in SIZE_ORDER else 2
    if direction == 'up':
        return SIZE_ORDER[min(idx + 1, len(SIZE_ORDER) - 1)]
    return SIZE_ORDER[max(idx - 1, 0)]

def get_all_size_options(current_size):
    """Return nearby sizes for display."""
    s = str(current_size).upper().strip()
    idx = SIZE_ORDER.index(s) if s in SIZE_ORDER else 2
    return {
        'down2': SIZE_ORDER[max(idx - 2, 0)],
        'down1': SIZE_ORDER[max(idx - 1, 0)],
        'current': s,
        'up1': SIZE_ORDER[min(idx + 1, 5)],
        'up2': SIZE_ORDER[min(idx + 2, 5)],
    }

def encode_image_for_claude(img_path, max_size=1024):
    """Encode image as base64 JPEG for Claude Vision API."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def measure_garment_dimensions(garment_img_path):
    """
    Estimate garment physical proportions from its image pixels.
    Returns a dict with width_ratio, height_ratio, aspect, and a size_hint.
    """
    try:
        img = Image.open(garment_img_path).convert('RGBA')
        arr = np.array(img)
        # Detect non-transparent / non-white pixels as garment pixels
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3]
            garment_mask = alpha > 30
        else:
            rgb = arr[:, :, :3].astype(np.float32)
            brightness = rgb.mean(axis=2)
            garment_mask = brightness < 240
        rows = np.any(garment_mask, axis=1)
        cols = np.any(garment_mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        g_h = rmax - rmin
        g_w = cmax - cmin
        ih, iw = arr.shape[:2]
        width_ratio  = g_w / iw if iw > 0 else 0.7
        height_ratio = g_h / ih if ih > 0 else 0.8
        aspect = g_w / g_h if g_h > 0 else 1.0
        # Heuristic: wide garments tend to be larger cuts
        if width_ratio > 0.85:
            size_hint = 'oversized'
        elif width_ratio > 0.70:
            size_hint = 'regular'
        else:
            size_hint = 'slim'
        return {
            'width_ratio':  round(width_ratio, 3),
            'height_ratio': round(height_ratio, 3),
            'aspect':       round(aspect, 3),
            'size_hint':    size_hint,
            'px_width':     int(g_w),
            'px_height':    int(g_h),
        }
    except Exception as e:
        print(f"[Garment measure] {e}")
        return None

def measure_body_from_image(user_img_path, gtype):
    """
    Estimate body width ratios from the user photo using pixel analysis.
    Returns shoulder_ratio, torso_ratio, hip_ratio, body_class.
    """
    try:
        img = Image.open(user_img_path).convert('RGB')
        arr = np.array(img)
        uh, uw = arr.shape[:2]
        # Background colour estimated from corners
        corners = np.array([arr[0,0], arr[0,-1], arr[10,0], arr[10,-1]], dtype=np.float32)
        bg = corners.mean(axis=0)

        def body_width_at(y_frac, band=0.04):
            y1 = int(uh * (y_frac - band))
            y2 = int(uh * (y_frac + band))
            y1, y2 = max(0, y1), min(uh, y2)
            strip = arr[y1:y2, :, :].astype(np.float32)
            col_bg_dist = np.sqrt(((strip - bg) ** 2).sum(axis=(0, 2)) / strip.shape[0])
            body_cols = col_bg_dist > 20
            # Measure the span (leftmost to rightmost body pixel)
            indices = np.where(body_cols)[0]
            if len(indices) < 5:
                return 0.5
            return (indices[-1] - indices[0]) / uw

        shoulder_r = body_width_at(0.25)
        chest_r    = body_width_at(0.32)
        waist_r    = body_width_at(0.45)
        hip_r      = body_width_at(0.58)

        # Body classification
        max_r = max(shoulder_r, chest_r, waist_r, hip_r)
        if max_r < 0.30:
            body_class = 'slim'
        elif max_r < 0.42:
            body_class = 'average'
        elif max_r < 0.52:
            body_class = 'athletic_or_curvy'
        else:
            body_class = 'plus'

        cfg  = PLACEMENT.get(gtype, PLACEMENT['default'])
        zone = cfg['zone']
        if zone == 'upper':
            key_r = max(shoulder_r, chest_r)
        elif zone == 'lower':
            key_r = max(waist_r, hip_r)
        else:
            key_r = max(shoulder_r, chest_r, waist_r, hip_r)

        return {
            'shoulder_ratio': round(shoulder_r, 3),
            'chest_ratio':    round(chest_r, 3),
            'waist_ratio':    round(waist_r, 3),
            'hip_ratio':      round(hip_r, 3),
            'key_ratio':      round(key_r, 3),
            'body_class':     body_class,
        }
    except Exception as e:
        print(f"[Body measure] {e}")
        return None

def analyze_fit_with_claude(user_img_path, garment_img_path, gtype, garment_size):
    """
    Professional size advisor using Claude Vision.
    Analyses body proportions + garment cut and returns a detailed JSON recommendation.
    """
    if not ANTHROPIC_API_KEY:
        return None
    try:
        user_b64    = encode_image_for_claude(user_img_path, max_size=1024)
        garment_b64 = encode_image_for_claude(garment_img_path, max_size=640)

        gs          = str(garment_size).upper().strip()
        size_info   = SIZE_MEASUREMENTS.get(gs, SIZE_MEASUREMENTS['M'])
        size_up     = get_size_suggestion(gs, 'up')
        size_down   = get_size_suggestion(gs, 'down')
        fit_guide   = GARMENT_FIT_GUIDE.get(gtype, GARMENT_FIT_GUIDE['default'])
        gdims       = measure_garment_dimensions(garment_img_path)
        gdim_note   = f"Garment image analysis: cut appears {gdims['size_hint']} (width ratio {gdims['width_ratio']:.0%} of frame)" if gdims else "Garment dimensions: unavailable"

        prompt = f"""You are a senior professional fashion stylist and fit consultant for VTRY, a premium virtual try-on platform.

You will receive TWO images:
  IMAGE 1 — the CLIENT's full-body or half-body photo
  IMAGE 2 — the GARMENT product photo

Your job: analyse the client's body and the garment, then deliver a precise, professional size recommendation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GARMENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type            : {gtype}
Tagged size     : {gs}  ({size_info['label']})
Standard chest  : {size_info['chest'][0]}–{size_info['chest'][1]} inches
Standard waist  : {size_info['waist'][0]}–{size_info['waist'][1]} inches
Standard hips   : {size_info['hips'][0]}–{size_info['hips'][1]} inches
Preferred fit   : {fit_guide['preferred']}
Key fit area    : {fit_guide['key_area']}
If too tight    : {fit_guide['tight_warn']}
Loose is OK?    : {'Yes — this style works oversized' if fit_guide['loose_ok'] else 'No — should be tailored'}
{gdim_note}
Available sizes : {', '.join(SIZE_ORDER)}
Next size up    : {size_up}
Next size down  : {size_down}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYSIS INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 — CLIENT BODY ANALYSIS
  • Estimate body frame: skinny / slim / average / athletic / curvy / plus-size / heavy
  • Note visible proportions: shoulder width, chest/bust, waist definition, hips
  • Judge overall silhouette — is the person lean, medium-built, or broad?

Step 2 — GARMENT CUT ANALYSIS
  • Look at the garment in image 2: is it cut slim, regular, or oversized?
  • Does the garment appear large/small relative to a standard {gs}?
  • Is the garment structured (like a blazer) or relaxed (like a hoodie)?

Step 3 — FIT ASSESSMENT
  • Compare the client's estimated measurements against size {gs}
  • For a SKINNY or very slim person wearing a large-cut garment: flag as too_loose and recommend sizing down
  • For a PLUS or broad client wearing a slim-cut garment tagged {gs}: flag as too_tight and recommend sizing up
  • For AVERAGE build with standard cut: assess whether {gs} is correct, too big, or too small
  • Always factor in garment type — hoodies and blouses allow more ease; shirts, jeans, and suits should be close-fitted

Step 4 — RECOMMENDATION
  • Give one primary recommended size
  • Give one alternative size
  • Be specific in your message — mention the body area driving the recommendation
  • If the garment looks oversized/wide even at {gs}, account for that

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — respond ONLY with this exact JSON, no extra text, no markdown fences:
{{
  "body_type": "slim",
  "build_description": "Slender frame with narrow shoulders and lean torso",
  "garment_cut": "oversized",
  "current_size_assessment": "too_loose",
  "recommended_size": "{size_down}",
  "alternative_size": "{gs}",
  "fit_label": "⚪ Too Loose — Size Down",
  "fit_color": "#9e9e9e",
  "fit_score": 30,
  "message": "The garment is cut wide and your slim frame will be lost in size {gs}. We recommend {size_down} for a cleaner silhouette.",
  "detailed_advice": "Focus on the shoulder seam — it should sit at the edge of your shoulder, not drooping. Size {size_down} will give you a much better drape across the chest without looking baggy.",
  "styling_tip": "If you prefer an oversized streetwear look, {gs} works — but for a polished appearance, go {size_down}.",
  "body_measurements_est": "Chest ~32 in, Waist ~26 in, Hips ~35 in",
  "confidence": "high"
}}

current_size_assessment options (pick ONE):
  "perfect"        → 🟢 Perfect Fit
  "slightly_tight" → 🟠 Slightly Tight — Size Up
  "too_tight"      → 🔴 Too Tight — Size Up
  "slightly_loose" → 🔵 Slightly Loose — Size Down
  "too_loose"      → ⚪ Too Loose — Size Down

fit_score: integer 0–100  (95–100 = perfect, 60–80 = acceptable, <50 = wrong size)
garment_cut options: "slim", "regular", "oversized", "relaxed", "structured"
confidence options: "high", "medium", "low"
"""

        payload = {
            "model": "claude-opus-4-5",
            "max_tokens": 900,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": user_b64}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": garment_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json=payload,
            timeout=45
        )

        if resp.status_code != 200:
            print(f"[Claude Fit] API error {resp.status_code}: {resp.text[:300]}")
            return None

        data = resp.json()
        raw  = data['content'][0]['text'].strip()
        raw  = raw.replace('```json', '').replace('```', '').strip()
        result = json.loads(raw)

        # Map assessment → display values
        ASSESS_MAP = {
            'perfect':        ('🟢 Perfect Fit',            '#00c853'),
            'slightly_tight': ('🟠 Slightly Tight — Size Up','#ff8c00'),
            'too_tight':      ('🔴 Too Tight — Size Up',     '#ff4444'),
            'slightly_loose': ('🔵 Slightly Loose — Size Down','#2196f3'),
            'too_loose':      ('⚪ Too Loose — Size Down',   '#9e9e9e'),
        }
        assessment = result.get('current_size_assessment', 'perfect')
        default_label, default_color = ASSESS_MAP.get(assessment, ('🟢 Perfect Fit', '#00c853'))

        gs_norm = str(garment_size).upper().strip()
        return {
            'status':               assessment,
            'label':                result.get('fit_label', default_label),
            'message':              result.get('message', ''),
            'color':                result.get('fit_color', default_color),
            'body_type':            result.get('body_type', 'unknown'),
            'build_description':    result.get('build_description', ''),
            'garment_cut':          result.get('garment_cut', 'regular'),
            'recommended_size':     result.get('recommended_size', gs_norm),
            'alternative_size':     result.get('alternative_size', gs_norm),
            'current_size':         gs_norm,
            'detailed_advice':      result.get('detailed_advice', ''),
            'styling_tip':          result.get('styling_tip', ''),
            'body_measurements_est':result.get('body_measurements_est', ''),
            'confidence':           result.get('confidence', 'medium'),
            'fit_score':            int(result.get('fit_score', 75)),
            'ai_fit_analysis':      True,
            'size_up':              get_size_suggestion(gs_norm, 'up'),
            'size_down':            get_size_suggestion(gs_norm, 'down'),
            'all_sizes':            get_all_size_options(gs_norm),
        }

    except Exception as e:
        print(f"[Claude Fit] Exception: {e}")
        return None


def analyze_fit_pixel(user_img_path, garment_img_path, gtype, garment_size):
    """
    Smart pixel-based fallback fit analysis.
    Measures body width and garment dimensions from raw pixels.
    """
    gs       = str(garment_size).upper().strip()
    size_up  = get_size_suggestion(gs, 'up')
    size_down= get_size_suggestion(gs, 'down')

    base = {
        'current_size':  gs,
        'size_up':       size_up,
        'size_down':     size_down,
        'all_sizes':     get_all_size_options(gs),
        'ai_fit_analysis': False,
        'confidence':    'medium',
        'styling_tip':   '',
        'body_measurements_est': '',
        'garment_cut':   'regular',
        'build_description': '',
    }

    try:
        body   = measure_body_from_image(user_img_path, gtype)
        gdims  = measure_garment_dimensions(garment_img_path)
        cfg    = PLACEMENT.get(gtype, PLACEMENT['default'])

        body_r = body['key_ratio'] if body else 0.50
        body_class = body['body_class'] if body else 'average'

        # Expected garment width ratio for this size (empirical scale)
        sm = {'XS': 0.86, 'S': 0.91, 'M': 1.00, 'L': 1.08, 'XL': 1.16, 'XXL': 1.24}
        base_gwr = cfg['w']
        egr  = base_gwr * sm.get(gs, 1.00)

        # Factor in garment's actual cut from pixel measurement
        if gdims:
            if gdims['size_hint'] == 'oversized':
                egr *= 1.12
            elif gdims['size_hint'] == 'slim':
                egr *= 0.90
            base['garment_cut'] = gdims['size_hint']

        ratio = egr / body_r if body_r > 0 else 1.0

        # Body class nudges
        if body_class == 'slim' and ratio > 1.20:
            ratio *= 1.08   # slim person → garment will look even bigger
        elif body_class == 'plus' and ratio < 1.05:
            ratio *= 0.88   # plus person → garment will feel tighter

        base['body_type']  = body_class
        base['body_ratio'] = round(body_r, 3)
        base['garment_ratio'] = round(egr, 3)
        base['ratio']      = round(ratio, 3)

        fit_guide = GARMENT_FIT_GUIDE.get(gtype, GARMENT_FIT_GUIDE['default'])

        if ratio < 0.87:
            return {**base, 'status': 'too_tight',
                    'label': '🔴 Too Tight — Size Up',
                    'fit_score': 15,
                    'color': '#ff4444',
                    'recommended_size': size_up,
                    'alternative_size': gs,
                    'message': f'This {gtype} will be too tight — it will {fit_guide["tight_warn"]}. Size {size_up} is strongly recommended.',
                    'detailed_advice': f'The {fit_guide["key_area"]} area is the most critical for this garment. Size {size_up} will give you the right ease of movement and a better silhouette.'}
        elif ratio < 0.95:
            return {**base, 'status': 'slightly_tight',
                    'label': '🟠 Slightly Tight — Consider Sizing Up',
                    'fit_score': 52,
                    'color': '#ff8c00',
                    'recommended_size': size_up,
                    'alternative_size': gs,
                    'message': f'Size {gs} may feel snug, especially around the {fit_guide["key_area"]}. Try {size_up} for a more comfortable fit.',
                    'detailed_advice': f'If you prefer a close, tailored look and the fabric has stretch, {gs} can work. Otherwise {size_up} will be more comfortable.'}
        elif ratio <= 1.13:
            return {**base, 'status': 'perfect',
                    'label': '🟢 Perfect Fit',
                    'fit_score': 94,
                    'color': '#00c853',
                    'recommended_size': gs,
                    'alternative_size': gs,
                    'message': f'Size {gs} is an excellent match for your body proportions and this {gtype}.',
                    'detailed_advice': f'The {fit_guide["key_area"]} measurements align well. You should have good freedom of movement with a clean silhouette.'}
        elif ratio <= 1.25:
            return {**base, 'status': 'slightly_loose',
                    'label': '🔵 Slightly Loose — Consider Sizing Down',
                    'fit_score': 58,
                    'color': '#2196f3',
                    'recommended_size': size_down if not fit_guide['loose_ok'] else gs,
                    'alternative_size': gs,
                    'message': f'Size {gs} will be a bit relaxed on your frame.' + (f' Try {size_down} for a more fitted look.' if not fit_guide['loose_ok'] else ' This works great for a casual, relaxed style.'),
                    'detailed_advice': ('This garment type is designed to be worn relaxed, so the extra room is intentional.'
                                        if fit_guide['loose_ok']
                                        else f'Size {size_down} will give a cleaner, more polished silhouette around the {fit_guide["key_area"]}.')}
        else:
            return {**base, 'status': 'too_loose',
                    'label': '⚪ Too Loose — Size Down',
                    'fit_score': 20,
                    'color': '#9e9e9e',
                    'recommended_size': size_down,
                    'alternative_size': gs,
                    'message': f'Size {gs} will be very baggy on you. Size {size_down} is recommended for a much better look.',
                    'detailed_advice': f'The garment will likely droop at the {fit_guide["key_area"]} area. Sizing down to {size_down} will give structure and shape to the outfit.'}

    except Exception as e:
        print(f"[Pixel Fit] Error: {e}")
        return {**base,
                'status': 'unknown', 'label': '⚪ Fit Unknown',
                'fit_score': 50, 'color': '#9e9e9e',
                'recommended_size': gs, 'alternative_size': gs,
                'message': 'Could not fully analyse fit from this image.',
                'detailed_advice': 'For best results, upload a clear full-body or half-body photo with good lighting.'}


def analyze_fit(user_img_path, garment_img_path, gtype, garment_size):
    """
    Master fit analysis orchestrator.
    Tries Claude Vision AI first; falls back to smart pixel analysis.
    Always returns a complete, display-ready fit dict.
    """
    # 1. Try Claude AI Vision
    if ANTHROPIC_API_KEY:
        ai_result = analyze_fit_with_claude(user_img_path, garment_img_path, gtype, garment_size)
        if ai_result:
            print(f"[Fit] ✅ Claude AI — body: {ai_result.get('body_type')}, cut: {ai_result.get('garment_cut')}, recommended: {ai_result.get('recommended_size')}")
            return ai_result
        print("[Fit] ⚠ Claude AI failed — using pixel fallback")

    # 2. Pixel fallback
    result = analyze_fit_pixel(user_img_path, garment_img_path, gtype, garment_size)
    print(f"[Fit] 📐 Pixel analysis — status: {result.get('status')}, recommended: {result.get('recommended_size')}")
    return result

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
    # ── Shirts ────────────────────────────────────────────────────────
    ('Blue Formal Shirt',        'shirt',  'Blue',   'M',  'formal-shirt-blue-formal-shirt-with-button-up-design-tE2ADHya_t.jpg'),
    ('White Formal Shirt',       'shirt',  'White',  'M',  'formal-shirt-shirt-white-shirt-long-sleeves-button-down-collar-lightweight-fabric-10hMqcqh_t.jpg'),
    # ── Hoodies ───────────────────────────────────────────────────────
    ('Blue Casual Hoodie',       'hoodie', 'Blue',   'M',  'hoodie-comfortable-blue-hoodie-for-casual-wear-Z2e1wS7v_t.jpg'),
    ('Yellow Casual Hoodie',     'hoodie', 'Yellow', 'M',  'hoodie-cozy-yellow-hoodie-for-casual-wear-Q2NsVgyS_t.jpg'),
    ('Gray Casual Hoodie',       'hoodie', 'Gray',   'M',  'hoodie-gray-hoodie-for-casual-wear-eN9T92h3_t.jpg'),
    ('Pink Casual Hoodie',       'hoodie', 'Pink',   'S',  'hoodie-pink-hoodie-for-casual-wear-R5CuUctG_t.jpg'),
    ('Red Casual Hoodie',        'hoodie', 'Red',    'M',  'hoodie-red-hoodie-for-casual-wear-0G9zkbB4_t.jpg'),
    # ── Frocks ────────────────────────────────────────────────────────
    ('Frock Style 1',            'frock',  'Multi',  'M',  'frock1.jpg'),
    ('Frock Style 2',            'frock',  'Multi',  'M',  'frock2.jpg'),
    # ── Blouses ───────────────────────────────────────────────────────
    ('Pink Bow Blouse',          'blouse', 'Pink',   'S',  'real_blouse_pink.png'),
    ('Floral Blouse',            'blouse', 'Floral', 'M',  'real_blouse_floral1.jpg'),
    ('3D Flower Blouse',         'blouse', 'Peach',  'M',  'real_blouse_floral2.jpg'),
    ('White Embroidery Blouse',  'blouse', 'White',  'S',  'real_blouse_embroidery.jpg'),
    ('White Rose Blouse',        'blouse', 'White',  'M',  'real_blouse_white_rose.jpg'),
    ('Pink Bow Sleeve Blouse',   'blouse', 'Pink',   'S',  'real_blouse_pink_bow.jpg'),
    ('Pink Ruffled Camisole',    'blouse', 'Pink',   'S',  'real_camisole_pink.jpg'),
    # ── Jeans ─────────────────────────────────────────────────────────
    ('Blue Stripe Jeans',        'jeans',  'Blue',   'M',  'real_jeans_blue_stripe.jpg'),
    ('Wide Leg Jeans',           'jeans',  'Blue',   'L',  'real_jeans_wide.png'),
    ('Ripped Blue Jeans',        'jeans',  'Blue',   'M',  'real_jeans_ripped.jpg'),
    ('Navy Blue Jeans',          'jeans',  'Navy',   'L',  'real_jeans_navy.jpg'),
    # ── Pants ─────────────────────────────────────────────────────────
    ('Black Skinny Pants',       'pants',  'Black',  'M',  'real_pants_black.png'),
    ('Beige Chino Pants',        'pants',  'Beige',  'M',  'real_pants_beige.jpg'),
    ('Grey Formal Pants',        'pants',  'Grey',   'M',  'real_pants_grey.jpg'),
    ('Colorful Floral Pants',    'pants',  'Multi',  'M',  'pants-colorful-floral-pattern-pants-3NTS00kH_t.jpg'),
    # ── Skirts ────────────────────────────────────────────────────────
    ('White Elegant Skirt',      'skirt',  'White',  'M',  'real_skirt_white.jpg'),
    ('Brown Pleated Skirt',      'skirt',  'Brown',  'M',  'skirt-brown-pleated-skirt-illustration-7LBGzLJ6_t.jpg'),
    ('Floral Printed Skirt',     'skirt',  'Multi',  'M',  'skirt-colorful-floral-printed-skirt-in-bohemian-style-hBrmQbJd_t.jpg'),
    ('Floral Pattern Skirt',     'skirt',  'Floral', 'S',  'skirt-floral-patterned-skirt-design-A5FPXCBi_t.jpg'),
    ('Peach Bow Skirt',          'skirt',  'Peach',  'S',  'skirt-peach-colored-skirt-with-bow-gqgC2LfX_t.jpg'),
    ('Orange Bow Skirt',         'skirt',  'Orange', 'S',  'skirt-stylish-orange-bow-skirt-illustration-jQyKU6vy_t.jpg'),
    # ── Suits ─────────────────────────────────────────────────────────
    ('Black Formal Suit',        'suit',   'Black',  'L',  'real_suit_black.jpg'),
    # ── Other ─────────────────────────────────────────────────────────
    ('Casual Top',               'top',    'Multi',  'M',  '11.png'),
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
        # Always run smart fit analysis regardless of try-on mode
        fit = analyze_fit(u_path, g_path, garment['type'], garment['size'] or 'M')
        if mode == "ai":
            fit['ai_tryon'] = True  # tag that FASHN was used for the visual

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