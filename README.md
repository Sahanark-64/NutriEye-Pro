NutriEye is an AI-based nutrition intelligence system that analyzes food items from images and provides instant nutritional information using computer vision and deep learning.

---

## 📌 Features

- Upload a food image and get instant nutrition analysis
- Displays Calories, Protein, Carbohydrates, Fat, and Vitamins
- Enter food weight (grams) to get scaled nutrition values
- Macronutrient breakdown donut chart
- Nutrition vs Daily Goal bar chart
- BMI Calculator with category and color coding
- Real-time AI food detection using EfficientNetB3
- Color-based fallback detection for fruits (grapes, banana, apple, orange)

---

## 🗂️ Project Structure

```
NutriEye-Pro/
├── backend/
│   ├── app.py              # Flask API server
│   ├── model.py            # AI food detection (EfficientNetB3)
│   ├── preprocess.py       # Image preprocessing (OpenCV + CLAHE)
│   ├── train_model.py      # Model training script (MobileNetV2)
│   ├── nutrition.csv       # Food nutrition database (per 100g)
│   └── food_model.h5       # Trained model (if available)
├── frontend/
│   ├── index.html          # Main analyze page
│   ├── dashboard.html      # Daily nutrition dashboard
│   ├── style.css           # UI styling
│   └── app.js              # Frontend logic
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Requirements

- Python 3.11 (TensorFlow does not support Python 3.12+)
- pip packages listed in `requirements.txt`

---

## 🚀 Installation & Setup

### Step 1 — Install Python 3.11

Download from: https://www.python.org/downloads/release/python-3119/

During installation, check **"Add Python 3.11 to PATH"**

### Step 2 — Create Virtual Environment

Open terminal in the project folder and run:

```bash
py -3.11 -m venv env
env\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install flask flask-cors pandas numpy opencv-python tensorflow Pillow
```

### Step 4 — Run the Project

```bash
cd backend
python app.py
```

### Step 5 — Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. User uploads a food image
2. OpenCV preprocesses the image (resize, CLAHE contrast enhancement)
3. EfficientNetB3 (pretrained on ImageNet) classifies the image
4. If confidence is low, color-based detection identifies fruits
5. Predicted food label is matched to the nutrition database (CSV)
6. Nutrition values are scaled by the entered weight (default 100g)
7. Results displayed with charts

---

## 🍽️ Supported Foods

| Food | Calories | Protein | Carbs | Fat |
|------|----------|---------|-------|-----|
| Apple | 52 | 0.3g | 14g | 0.2g |
| Banana | 96 | 1.3g | 27g | 0.3g |
| Grapes | 69 | 0.7g | 18g | 0.2g |
| Orange | 47 | 0.9g | 12g | 0.1g |
| Pizza | 266 | 11g | 33g | 10g |
| Burger | 295 | 17g | 30g | 12g |
| Rice | 130 | 2.7g | 28g | 0.3g |
| Egg | 155 | 13g | 1.1g | 11g |
| Chicken | 239 | 27g | 0g | 14g |
| Steak | 271 | 26g | 0g | 18g |
| Pasta | 131 | 5g | 25g | 1.1g |
| Salad | 15 | 1.3g | 2.9g | 0.2g |
| Bread | 265 | 9g | 49g | 3.2g |
| Soup | 50 | 3g | 8g | 1g |
| Sushi | 150 | 9g | 18g | 4g |
| Sandwich | 250 | 12g | 30g | 9g |
| Hotdog | 290 | 10g | 24g | 18g |
| Donut | 452 | 5g | 51g | 25g |
| Ice Cream | 207 | 3.5g | 24g | 11g |
| French Fries | 312 | 3.4g | 41g | 15g |
| Chocolate Cake | 371 | 5g | 52g | 17g |

*All values are per 100g*

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Upload image file, returns nutrition data |
| POST | `/webcam` | Send base64 image, returns nutrition data |
| POST | `/bmi` | Send weight + height, returns BMI |
| GET | `/foods` | Returns all food names in database |

---

## 🏋️ Train Your Own Model (Optional)

For better accuracy, train on the Food-101 dataset:

1. Download dataset from: https://www.kaggle.com/datasets/dansbecker/food-101
2. Organize into `backend/dataset/train/` and `backend/dataset/val/`
3. Run:
```bash
cd backend
python train_model.py
```
4. Model saves as `backend/food_model.h5`

---

## 🔮 Future Improvements

- Train custom model on Food-101 for higher accuracy
- Add meal history and weekly nutrition tracking
- Integrate barcode scanner for packaged foods
- Add portion size estimation from image
- Mobile app version
- Multi-food detection in a single image

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| AI Model | TensorFlow, Keras, EfficientNetB3 |
| Image Processing | OpenCV, NumPy |
| Data | Pandas, CSV |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |

---

## 📄 License

This project is built for educational purposes.
