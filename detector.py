"""
Object Detection Pipeline (YOLOv8) + Pillow Sinhala/Tamil Label Rendering.
"""

from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from labels import get_native_label

# Model singleton cache
_MODEL = None

def get_yolo_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8n.pt")
    return _MODEL

def detect_and_count(pil_img: Image.Image, lang_code: str, conf_thresh: float = 0.40):
    """
    YOLOv8 මඟින් වස්තු හඳුනාගෙන Native Language Labels සාදයි.
    """
    model = get_yolo_model()
    # Ultralytics accepts PIL image directly
    results = model.predict(source=pil_img, conf=conf_thresh, iou=0.5, verbose=False)[0]
    
    counts_per_class = defaultdict(int)
    detected_objects = []
    
    # Process detected boxes
    if results.boxes is not None and len(results.boxes) > 0:
        boxes_data = results.boxes.data.cpu().numpy()
        for row in boxes_data:
            x1, y1, x2, y2, conf, cls_id = row
            cls_name = model.names[int(cls_id)]
            native_label = get_native_label(cls_name, lang_code)
            
            counts_per_class[native_label] += 1
            curr_index = counts_per_class[native_label]
            
            detected_objects.append({
                "label": native_label,
                "index": curr_index,
                "confidence": round(float(conf), 2),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
            
    total_objects = len(detected_objects)
    return total_objects, dict(counts_per_class), detected_objects

def draw_native_labels(pil_img: Image.Image, objects_list: list, font_path: str = "NotoSansSinhala-Regular.ttf") -> Image.Image:
    """
    Pillow මඟින් සිංහල / දෙමළ අකුරු කැඩෙන්නේ නැතිව ලස්සනට පින්තූරය මත අඳියි.
    """
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Load Font with fallback to default
    try:
        font = ImageFont.truetype(font_path, size=22)
    except Exception:
        font = ImageFont.load_default()
        
    for item in objects_list:
        x1, y1, x2, y2 = item["box"]
        label_text = f"{item['label']} {item['index']}"
        
        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline="#00FF66", width=3)
        
        # Background box for readable text
        bbox = draw.textbbox((x1, max(0, y1 - 28)), label_text, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill="#00AA44")
        draw.text((x1, max(0, y1 - 28)), label_text, fill="white", font=font)
        
    return annotated
