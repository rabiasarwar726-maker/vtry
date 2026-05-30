# VTRY — Virtual Try-On System
**Final Year Project | Rabia Sarwar (2022-ag-9162) | UAF**

---

## ⚡ Quick Setup (3 Steps)

### 1. Install dependencies
```bash
cd vtry
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## 🔑 Login Credentials

| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |
| User  | Register via /register |

---

## 📁 Project Structure

```
vtry/
├── app.py                  ← Main Flask application
├── requirements.txt        ← Python dependencies
├── instance/
│   └── vtry.db            ← SQLite database (auto-created)
├── static/
│   ├── uploads/           ← User uploaded photos
│   ├── garments/          ← Garment images
│   └── results/           ← Generated try-on results
└── templates/
    ├── base.html          ← Base layout
    ├── index.html         ← Landing page
    ├── login.html         ← User login
    ├── register.html      ← User registration
    ├── dashboard.html     ← User dashboard
    ├── catalog.html       ← Garment catalog
    ├── tryon.html         ← Try-on page
    ├── result.html        ← Result display
    ├── my_results.html    ← User results history
    ├── admin_login.html   ← Admin login
    ├── admin_dashboard.html ← Admin overview
    ├── admin_garments.html  ← Manage garments
    └── admin_users.html     ← View users
```

---

## 🛠 Tech Stack
- **Backend:** Python, Flask, SQLite
- **Image Processing:** OpenCV, NumPy, Pillow
- **Frontend:** HTML5, CSS3, JavaScript
- **Architecture:** Client-Server (MVC pattern),fashion.ai Api

---

## 🪄 How Virtual Try-On Works
1. User uploads a front-facing photo (JPG/PNG)
2. User selects a garment from the catalog
3. OpenCV reads and preprocesses both images
4. Garment is resized to fit upper-body region (75% width, 45% height)
5. PIL composites the garment overlay with transparency (alpha blending)
6. Result image is saved and displayed to the user

---

## 📋 Features
- ✅ User Registration & Secure Login (hashed passwords)
- ✅ Image Upload (JPG/PNG, max 16MB)
- ✅ Garment Catalog (browse & select)
- ✅ AI Virtual Try-On (OpenCV + PIL)
- ✅ Result Display & Download
- ✅ Try-On History (per user)
- ✅ Admin Dashboard (stats, manage garments & users)
- ✅ SQLite Database (Users, Garments, Results, Admins)

---

## 🔮 Future Improvements
- Deep learning body pose estimation (MediaPipe/OpenPose)
- Real-time webcam try-on
- Mobile app version
