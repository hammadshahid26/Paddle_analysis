# 🎾 Tennis Analytics Pipeline (Local Setup)

A complete pipeline for **tennis video analysis**, designed for performance tracking and coaching insights.

---

## 🚀 Features

- 🎾 Ball Tracking  
- 🧍 Player Detection  
- 📍 Bounce Detection  
- 🏸 Stroke Classification  
- 🗺️ Court Zoning  
- 📊 Coaching Report Generation  

---

## 📁 Project Structure

```
Paddle1/
│
├── main.py
├── ball_detector.py
├── bounce_detector.py
├── person_detector.py
├── shot_classifier.py
├── court_zones.py
├── court_mask.py
├── tracknet.py
├── one_euro.py
├── utils.py
├── coaching_llm.py
│
├── test_30s.mp4
├── yolo11m.pt
│
├── models/
│   ├── model_best.pt
│   └── ctb_regr_bounce.cbm
```

---

## ⚙️ Installation (Local Machine)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/hammadshahid26/Paddle_analysis.git
cd Paddle1
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

#### ▶️ Activate Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py \
  --path_ball_track_model models/model_best.pt \
  --path_bounce_model models/ctb_regr_bounce.cbm \
  --path_input_video test_30s.mp4 \
  --path_output_video output.mp4 \
  --path_shot_csv shots.csv \
  --path_zones_csv zones.csv \
  --path_coaching_report coaching.txt
```

---

## 📤 Outputs

| File | Description |
|------|------------|
| 🎥 `output.mp4` | Processed video with annotations |
| 📄 `shots.csv` | Detected stroke events |
| 📊 `zones.csv` | Court zone analysis |
| 🧠 `coaching.txt` | AI-generated coaching insights |

---

## 🧠 Models Used

| Model | Purpose |
|------|--------|
| `model_best.pt` | Ball Tracking |
| `ctb_regr_bounce.cbm` | Bounce Detection |
| `yolo11m.pt` | Player Detection |

---

## 📍 Bounce Detection Model (CatBoost)

The **CatBoostRegressor** model is used to predict ball bounces based on the trajectory obtained from ball tracking.

### 📥 Pretrained Models

- https://drive.google.com/file/d/1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ/view?usp=drive_link  
- https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing  

👉 Download and place them inside the `/models` directory.

---

## ⚠️ Notes

- ✅ Use **relative paths** (avoid `/content/` if running locally)  
- ⚡ GPU is recommended but optional  
- 🔥 For best performance, install **CUDA-enabled PyTorch**  
