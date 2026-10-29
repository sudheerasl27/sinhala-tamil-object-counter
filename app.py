"""
Streamlit Web App: Object Counting & Native-Language Labelling
All UI elements adhere strictly to Sinhala or Tamil (Zero English leak).
"""

import io
import json
import streamlit as st
from PIL import Image
import numpy as np

from labels import UI_TEXT
from image_utils import (
    fix_orientation_and_resize,
    calculate_hashes,
    check_duplicate,
    assess_and_enhance_quality
)
from detector import detect_and_count, draw_native_labels

# Session State for History & Cache
if "history" not in st.session_state:
    st.session_state["history"] = {}

st.set_page_config(page_title="Object Counter", layout="wide")

# 1. භාෂාව තේරීම (R3 Requirement)
selected_lang_name = st.sidebar.radio("භාෂාව / மொழி", ("සිංහල", "தமிழ்"))
lang_code = "si" if selected_lang_name == "සිංහල" else "ta"
t = UI_TEXT[lang_code]

st.title(t["title"])

# 2. Input ක්‍රමය තේරීම (Camera vs Upload - R1 & R2)
input_mode = st.radio(t["input_mode"], (t["upload_option"], t["cam_option"]))

uploaded_file = None
if input_mode == t["upload_option"]:
    uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input(t["cam_capture"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    
    # ගොනුව නිවැරදිදැයි පරීක්ෂාව
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
    except Exception:
        st.error(t["bad_file_type"])
        st.stop()
        
    # Preprocessing
    pil_image = fix_orientation_and_resize(pil_image)
    
    # Hashes for Duplicate Detection
    md5_hash, phash_val = calculate_hashes(raw_bytes, pil_image)
    is_dup, orig_key = check_duplicate(md5_hash, phash_val, st.session_state["history"])
    
    # Process Quality Check (Brightness & Blur)
    cv_bgr = np.array(pil_image)[:, :, ::-1]
    cv_processed, quality_meta, warnings = assess_and_enhance_quality(cv_bgr, lang_code)
    processed_pil = Image.fromarray(cv_processed[:, :, ::-1])
    
    # Duplicate Warning
    if is_dup:
        warnings.append(f"{t['duplicate_warning']} ({orig_key})")
        
    # Object Detection
    total_count, counts_dict, detected_objects = detect_and_count(
        processed_pil, lang_code=lang_code, conf_thresh=0.40
    )
    
    # Render Labels on Image using Sinhala / Tamil TrueType font
    font_file = "NotoSansSinhala-Regular.ttf" if lang_code == "si" else "NotoSansTamil-Regular.ttf"
    annotated_img = draw_native_labels(processed_pil, detected_objects, font_path=font_file)
    
    # Display Output
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image(annotated_img, use_container_width=True)
        
    with col2:
        # Warnings
        if warnings:
            st.warning("⚠️ " + " | ".join(warnings))
            
        # Report Counts (R5)
        if total_count == 0:
            st.info(f"ℹ️ {t['no_objects']}")
        else:
            counts_str = ", ".join([f"{k} {v}" for k, v in counts_dict.items()])
            st.success(f"**{t['total_count']}: {total_count}** — {counts_str}")
            
        # Build Required JSON (R8)
        result_json = {
            "language": lang_code,
            "total_objects": total_count,
            "counts": counts_dict,
            "objects": detected_objects,
            "image_quality": quality_meta,
            "warnings": warnings,
            "duplicate_of": orig_key if is_dup else None
        }
        
        # Save to history for future duplicate checking
        if not is_dup:
            st.session_state["history"][md5_hash[:8]] = {
                "md5": md5_hash,
                "phash": phash_val,
                "result": result_json
            }
            
        st.subheader("JSON Output")
        st.json(result_json)
        
        # Download buttons
        img_buffer = io.BytesIO()
        annotated_img.save(img_buffer, format="JPEG")
        st.download_button(
            label=t["download_img"],
            data=img_buffer.getvalue(),
            file_name="labelled_output.jpg",
            mime="image/jpeg"
        )
        st.download_button(
            label=t["download_json"],
            data=json.dumps(result_json, ensure_ascii=False, indent=2),
            file_name="result.json",
            mime="application/json"
        )
