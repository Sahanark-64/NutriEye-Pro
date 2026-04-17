"""
model.py - Food detection using EfficientNetB3 pretrained on Food-101 classes.

Detection chain:
  1. EfficientNetB3 (ImageNet) - high confidence matches
  2. Color-based detection - fruits not in ImageNet
  3. Unknown fallback
"""

import cv2
import numpy as np

_model = None

# All foods in nutrition.csv mapped from possible ImageNet/model labels
LABEL_MAP = {
    # Apple
    "apple": "apple",
    "granny_smith": "apple",
    # Banana
    "banana": "banana",
    # Orange
    "orange": "orange",
    "lemon": "orange",
    "lime": "orange",
    "tangerine": "orange",
    "clementine": "orange",
    # Pizza
    "pizza": "pizza",
    # Burger
    "cheeseburger": "burger",
    "hamburger": "burger",
    # Hotdog
    "hotdog": "hotdog",
    "hot_dog": "hotdog",
    "corn_dog": "hotdog",
    # Bread
    "french_loaf": "bread",
    "bagel": "bread",
    "pretzel": "bread",
    "bun": "bread",
    "loaf": "bread",
    # Rice
    "rice": "rice",
    "pilaf": "rice",
    # Egg
    "egg": "egg",
    "omelette": "egg",
    "fried_egg": "egg",
    # Chicken
    "hen": "chicken",
    "cock": "chicken",
    "drumstick": "chicken",
    "fried_chicken": "chicken",
    "chicken": "chicken",
    # Salad
    "head_cabbage": "salad",
    "broccoli": "salad",
    "cauliflower": "salad",
    "mushroom": "salad",
    "guacamole": "salad",
    "salad": "salad",
    # Pasta
    "carbonara": "pasta",
    "spaghetti": "pasta",
    "noodle": "pasta",
    "pasta": "pasta",
    # Sandwich
    "submarine_sandwich": "sandwich",
    "burrito": "sandwich",
    "sandwich": "sandwich",
    # Sushi
    "sushi": "sushi",
    # Steak
    "steak": "steak",
    "meat_loaf": "steak",
    "sirloin": "steak",
    # Soup
    "soup_bowl": "soup",
    "consomme": "soup",
    "chowder": "soup",
    # Hotdog already above
    # Donut
    "doughnut": "donut",
    "donut": "donut",
    # Ice cream
    "ice_cream": "ice_cream",
    "ice_lolly": "ice_cream",
    "sundae": "ice_cream",
    # French fries
    "french_fries": "french_fries",
    "french_loaf": "bread",
    # Chocolate cake
    "chocolate_sauce": "chocolate_cake",
    "birthday_cake": "chocolate_cake",
    "chocolate_cake": "chocolate_cake",
    "cake": "chocolate_cake",
}


def _load_model():
    global _model
    if _model is not None:
        return
    import tensorflow as tf
    _model = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=True,
        input_shape=(300, 300, 3)
    )
    print("[MODEL] EfficientNetB3 loaded")


def _color_detect(image_path):
    """
    Detect fruits by color analysis after removing background.
    Returns (food_name, score) or (None, 0).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, 0

    img = cv2.resize(img, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Remove white/light background aggressively
    bg    = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    # Also remove near-grey pixels
    grey  = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 25, 255]))
    bg    = cv2.bitwise_or(bg, grey)
    fg    = cv2.bitwise_not(bg)
    # Erode to remove background edges
    kernel = np.ones((3,3), np.uint8)
    fg     = cv2.erode(fg, kernel, iterations=2)
    total  = max(int(np.sum(fg > 0)), 1)

    def pct(mask):
        return float(np.sum(cv2.bitwise_and(mask, fg) > 0)) / total

    # ── Grapes ───────────────────────────────────────────────────────────────
    # Red/black grapes: dark purple
    purple     = cv2.inRange(hsv, np.array([110, 30, 15]), np.array([165, 255, 180]))
    # Green grapes: yellow-green, MUTED saturation (30-150), not vivid yellow
    grn_grape  = cv2.inRange(hsv, np.array([38,  25, 50]), np.array([80,  150, 210]))
    grape_s    = pct(purple) + pct(grn_grape)

    # ── Banana ───────────────────────────────────────────────────────────────
    # Very vivid warm yellow only — saturation >180, grapes can never match this
    banana_m   = cv2.inRange(hsv, np.array([20, 180, 170]), np.array([33, 255, 255]))
    banana_s   = pct(banana_m)

    # ── Apple ────────────────────────────────────────────────────────────────
    red1       = cv2.inRange(hsv, np.array([0,   120, 60]), np.array([8,  255, 255]))
    red2       = cv2.inRange(hsv, np.array([172, 120, 60]), np.array([180,255, 255]))
    apple_s    = pct(red1) + pct(red2)

    # ── Orange ───────────────────────────────────────────────────────────────
    orange_m   = cv2.inRange(hsv, np.array([8, 160, 150]), np.array([18, 255, 255]))
    orange_s   = pct(orange_m)

    scores = {
        "grapes": grape_s,
        "banana": banana_s,
        "apple":  apple_s,
        "orange": orange_s,
    }
    print(f"[COLOR] {  {k: round(v,3) for k,v in scores.items()}  }")

    best  = max(scores, key=scores.get)
    score = scores[best]

    # Grapes needs lower threshold (muted color)
    threshold = 0.02 if best == "grapes" else 0.05
    if score >= threshold:
        return best, score
    return None, 0


def predict_food(image_path):
    from preprocess import preprocess_image
    import tensorflow as tf

    _load_model()

    img    = preprocess_image(image_path)
    img_in = tf.keras.applications.efficientnet.preprocess_input(img * 255.0)
    preds  = _model.predict(img_in, verbose=0)
    decoded = tf.keras.applications.efficientnet.decode_predictions(preds, top=10)[0]

    print("[MODEL] Top-10:")
    for _, lbl, sc in decoded:
        print(f"  {lbl}: {round(float(sc)*100,1)}%")

    # ── Step 1: strong ImageNet match (>35% confidence) ──────────────────────
    for _, lbl, sc in decoded:
        if float(sc) < 0.35:
            break
        key = lbl.lower().replace("-", "_")
        if key in LABEL_MAP:
            food = LABEL_MAP[key]
            print(f"[MATCH] {food} via '{lbl}' ({round(float(sc)*100,1)}%)")
            return {"food": food, "confidence": round(float(sc)*100, 1)}
        for k, v in LABEL_MAP.items():
            if k in key or key in k:
                print(f"[MATCH] {v} via '{lbl}' ({round(float(sc)*100,1)}%)")
                return {"food": v, "confidence": round(float(sc)*100, 1)}

    # ── Step 2: color detection (fruits) ─────────────────────────────────────
    color_food, color_score = _color_detect(image_path)
    if color_food:
        top_conf = round(float(decoded[0][2])*100, 1)
        print(f"[COLOR MATCH] {color_food}")
        return {"food": color_food, "confidence": top_conf}

    # ── Step 3: any ImageNet match regardless of confidence ───────────────────
    for _, lbl, sc in decoded:
        key = lbl.lower().replace("-", "_")
        if key in LABEL_MAP:
            return {"food": LABEL_MAP[key], "confidence": round(float(sc)*100, 1)}
        for k, v in LABEL_MAP.items():
            if k in key or key in k:
                return {"food": v, "confidence": round(float(sc)*100, 1)}

    print("[MODEL] No food detected.")
    return {"food": "unknown", "confidence": 0}
