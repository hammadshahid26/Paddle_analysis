# 🎾 Tennis Analytics Pipeline (Local Setup)

This project performs **tennis video analysis** including:

-   Ball tracking\
-   Player detection\
-   Bounce detection\
-   Stroke classification\
-   Court zoning\
-   Coaching report generation

------------------------------------------------------------------------

## 📁 Project Structure

Paddle1/

├── main.py\
├── ball_detector.py\
├── bounce_detector.py\
├── person_detector.py\
├── shot_classifier.py\
├── court_zones.py\
├── court_mask.py\
├── tracknet.py\
├── one_euro.py\
├── utils.py\
├── coaching_llm.py

├── test_30s.mp4\
├── yolo11m.pt

├── models/\
│ ├── model_best.pt\
│ └── ctb_regr_bounce.cbm

------------------------------------------------------------------------

## ⚙️ Installation (Local Machine)

### 1. Clone / Download Project

git clone <https://github.com/hammadshahid26/Paddle_analysis.git>
cd Paddle1

------------------------------------------------------------------------

### 2. Create Virtual Environment (Recommended)

python -m venv venv

Windows:\
venv`\Scripts`{=tex}`\activate  `{=tex}

Linux/Mac:\
source venv/bin/activate

------------------------------------------------------------------------

### 3. Install Dependencies

  pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Run the Project

python main.py\
--path_ball_track_model models/model_best.pt\
--path_bounce_model models/ctb_regr_bounce.cbm\
--path_input_video test_30s.mp4\
--path_output_video output.mp4\
--path_shot_csv shots.csv\
--path_zones_csv zones.csv\
--path_coaching_report coaching.txt

------------------------------------------------------------------------

## 📤 Outputs

-   output.mp4 → Processed video\
-   shots.csv → Stroke events\
-   zones.csv → Court zones\
-   coaching.txt → Coaching insights

------------------------------------------------------------------------

## 🧠 Models Used

-   model_best.pt → Ball Tracking\
-   ctb_regr_bounce.cbm → Bounce Detection\
-   yolo11m.pt → Player Detection

------------------------------------------------------------------------

## ⚠️ Notes

-   Use relative paths (no /content/)\
-   GPU recommended but optional\
-   Install CUDA-enabled PyTorch for GPU

------------------------------------------------------------------------

