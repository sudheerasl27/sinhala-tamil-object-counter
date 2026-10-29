### Suggested Repository Names

* `sinhala-tamil-object-counter`
* `native-vision-counter`
* `indic-object-detection`

---

### Repository Description

> A robust computer vision application that detects and counts objects in images with strict native Sinhala and Tamil UI rendering, edge-case failure handling, and structured JSON output.

---

### `README.md`

```markdown
# Native-Language Object Detection & Counting System

A robust computer vision application designed to capture or upload images, perform object detection, accurately count items, and report all labels, counts, warnings, and UI elements exclusively in **Sinhala (සිංහල)** or **Tamil (தமிழ்)**.

---

## Key Features

- **Strict Native Script Rendering:** Displays complex Unicode Sinhala and Tamil characters correctly using Pillow and Noto Sans fonts without glyph breaks or question mark (`?`) placeholders.
- **Zero English Leak:** UI elements, counts, bounding box annotations, warnings, and fallback labels strictly comply with the user-selected language.
- **Robust Edge Case Handling:**
  - **Duplicate Detection:** Exact match via MD5 hashing and near-duplicate detection via Perceptual Hashing (`pHash`).
  - **Lighting & Quality Assessment:** Laplacian variance check for motion blur, brightness calculations, and CLAHE enhancement for underexposed inputs.
  - **Input Normalization:** Auto-adjusts EXIF phone orientation and scales images exceeding maximum bounds.
  - **Fallback Safety:** Unregistered object classes safely map to native fallback labels instead of leaking raw English model names.
- **Standardized Export:** Outputs annotated images alongside complete JSON reports containing counts, bounding box coordinates, quality metrics, and duplicate history references.

---

## Project Structure

```text
├── app.py                 # Streamlit web application and UI pipeline
├── detector.py            # YOLOv8 inference, NMS, and PIL Unicode rendering
├── image_utils.py         # Quality checks (blur/brightness), CLAHE, and hash logic
├── labels.py              # Dictionary translations and multilingual UI strings
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/sinhala-tamil-object-counter.git
cd sinhala-tamil-object-counter

```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Font Configuration

Download the required TrueType fonts from Google Fonts and place them in the root directory:

* `NotoSansSinhala-Regular.ttf`
* `NotoSansTamil-Regular.ttf`

---

## Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

---

## JSON Output Schema

The system automatically generates structured outputs following this format:

```json
{
  "language": "si",
  "total_objects": 3,
  "counts": {
    "පුටුව": 2,
    "බෝතලය": 1
  },
  "objects": [
    {
      "label": "පුටුව",
      "index": 1,
      "confidence": 0.88,
      "box": [120, 340, 260, 520]
    }
  ],
  "image_quality": {
    "brightness": 0.21,
    "blur_score": 48.2,
    "enhanced": true
  },
  "warnings": [
    "අඳුරු ඡායාරූපයකි — ප්‍රතිඵල නිවැරදි නොවිය හැක"
  ],
  "duplicate_of": null
}

```

---

## Tech Stack

* **Framework:** Streamlit
* **Detection Model:** YOLOv8 Nano (`ultralytics`)
* **Image Processing:** OpenCV (`opencv-python-headless`), Pillow (`PIL`)
* **Deduplication:** `imagehash`, `hashlib`

```

<FollowUp>
Would you also like a `.gitignore` template or sample commands to push this directly to your new GitHub repository?
</FollowUp>

```
