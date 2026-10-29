"""
පින්තූර පෙර සැකසුම (Preprocessing), ගුණාත්මකභාවය (Quality Check),
සහ පිටපත් හඳුනාගැනීම (Duplicate Detection).
"""

import hashlib
import cv2
import numpy as np
from PIL import Image, ImageOps
import imagehash

def fix_orientation_and_resize(pil_img: Image.Image, max_side: int = 1280) -> Image.Image:
    """EXIF orientation හරිගස්සා උපරිම පැත්ත 1280px වන සේ resize කරයි."""
    # EXIF tags කියවා හරියටම rotate කිරීම
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return pil_img

def calculate_hashes(raw_bytes: bytes, pil_img: Image.Image):
    """Exact duplicate (MD5) සහ Near duplicate (Perceptual Hash) ගණනය කරයි."""
    md5_hash = hashlib.md5(raw_bytes).hexdigest()
    phash_val = str(imagehash.phash(pil_img))
    return md5_hash, phash_val

def check_duplicate(current_md5: str, current_phash: str, history: dict, phash_thresh: int = 6):
    """
    පෙර සැකසූ පින්තූර සමඟ සසඳා duplicate එකක්දැයි බලයි.
    Returns: (is_duplicate: bool, original_key: str or None)
    """
    for item_id, data in history.items():
        # Exact match
        if current_md5 == data["md5"]:
            return True, item_id
        # Near match using Hamming distance on phash
        prev_phash = imagehash.hex_to_hash(data["phash"])
        curr_ph = imagehash.hex_to_hash(current_phash)
        if (curr_ph - prev_phash) <= phash_thresh:
            return True, item_id
    return False, None

def assess_and_enhance_quality(cv_img_bgr, lang_code: str):
    """
    Brightness සහ Blur ලකුණු ගණනය කරයි.
    අඳුරු නම් CLAHE මඟින් contrast වැඩි කරයි.
    Warnings අදාළ දේශීය භාෂාවෙන් ජනනය කරයි.
    """
    gray = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Blur score using Variance of Laplacian
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Brightness (0.0 to 1.0)
    brightness = float(np.mean(gray) / 255.0)
    
    warnings = []
    enhanced = False
    processed_bgr = cv_img_bgr.copy()
    
    # Low light handling
    if brightness < 0.22:
        enhanced = True
        # CLAHE on L-channel of LAB
        lab = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        warn_msg = (
            "අඳුරු ඡායාරූපයකි — ප්‍රතිඵල නිවැරදි නොවිය හැක"
            if lang_code == "si"
            else "இருண்ட படம் — முடிவுகள் துல்லியமாக இருக்காது"
        )
        warnings.append(warn_msg)
    elif brightness > 0.85:
        warn_msg = (
            "අධික ආලෝකය සහිත ඡායාරූපයකි"
            if lang_code == "si"
            else "அதிக பிரகாசமான படம்"
        )
        warnings.append(warn_msg)
        
    # Blur detection
    if blur_score < 45.0:
        warn_msg = (
            "ඡායාරූපය බොඳ වී ඇත — නැවත පැහැදිලි ඡායාරූපයක් ලබාගන්න"
            if lang_code == "si"
            else "படம் மங்கலாக உள்ளது — மீண்டும் தெளிவான படம் எடுக்கவும்"
        )
        warnings.append(warn_msg)
        
    quality_meta = {
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 1),
        "enhanced": enhanced
    }
    return processed_bgr, quality_meta, warnings
