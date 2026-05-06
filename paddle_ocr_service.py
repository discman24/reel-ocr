# REEL — PaddleOCR Microservice
# Deploy on Railway (or Render). Costs ~$5/month.
# Set OCR_SERVICE_URL=https://your-service.railway.app in Vercel env.
#
# Scans 5 image regions:
#   full       — all text in frame
#   subtitle   — bottom-center (burned-in subtitles, near-unique scene fingerprint)
#   bottom_left  — Douyin/Kuaishou watermarks (抖音号, creator ID)
#   bottom_right — ReelShort / DramaBox / WeTV platform logos
#   top_overlay  — show title cards, episode numbers
#
# Runs Chinese (PP-OCRv4) + Korean models on every region. Merges unique results.
# Confidence threshold: 0.7 (drops noise from blurry/unclear images).

from paddleocr import PaddleOCR
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import cv2
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paddle_ocr_service")

app = FastAPI(title="REEL OCR Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize models at startup — takes 5-10s, then warm for all subsequent requests.
# lang='ch' covers Simplified Chinese + English.
# lang='korean' covers Hangul (Korean script).
# Both run on CPU by default. GPU available on paid Railway tiers.
logger.info("Initializing PaddleOCR models...")
ocr_ch = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
ocr_ko = PaddleOCR(use_angle_cls=True, lang="korean", show_log=False)
logger.info("PaddleOCR models ready.")


def extract_region_text(region_img, confidence_threshold=0.7):
    """Run both OCR models on a region, merge unique results above confidence threshold."""
    texts = []
    for model in [ocr_ch, ocr_ko]:
        try:
            result = model.ocr(region_img, cls=True)
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence >= confidence_threshold and len(text.strip()) >= 1:
                        texts.append(text.strip())
        except Exception as e:
            logger.warning(f"OCR model error: {e}")
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


@app.get("/health")
async def health():
    """Keep-alive endpoint. Ping this every 5 minutes to prevent cold starts."""
    return {"status": "ok", "models": ["ch", "korean"]}


@app.post("/ocr")
async def extract_text(payload: dict):
    """
    POST /ocr
    Body: { "image": "<base64-encoded image>" }
    Returns: {
        "regions": { "full": [...], "subtitle_band": [...], ... },
        "all_text": [...],   # flattened, deduplicated across all regions
        "subtitle": [...],   # subtitle_band results only
        "watermarks": [...]  # bottom_left + bottom_right results
    }
    """
    try:
        img_bytes = base64.b64decode(payload["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image", "all_text": [], "subtitle": [], "watermarks": []}

        h, w = img.shape[:2]

        # Define 5 scan regions — tuned for Asian drama app screenshots
        regions = {
            "full": img,
            # Subtitle band: center-bottom 15%, offset 10% from sides (avoids edge noise)
            "subtitle_band": img[int(h * 0.82):h, int(w * 0.1):int(w * 0.9)],
            # Bottom-left: Douyin 抖音号 watermarks, creator IDs
            "bottom_left": img[int(h * 0.85):h, 0:int(w * 0.35)],
            # Bottom-right: ReelShort, DramaBox, WeTV, iQIYI platform logos
            "bottom_right": img[int(h * 0.85):h, int(w * 0.65):w],
            # Top overlay: show title cards, episode numbers, network branding
            "top_overlay": img[0:int(h * 0.15), :],
        }

        results = {}
        for region_name, region_img in regions.items():
            if region_img.size == 0:
                results[region_name] = []
                continue
            results[region_name] = extract_region_text(region_img)

        # Flatten all text, deduplicated
        seen = set()
        all_text = []
        for texts in results.values():
            for t in texts:
                if t not in seen:
                    seen.add(t)
                    all_text.append(t)

        watermarks = list(set(results.get("bottom_left", []) + results.get("bottom_right", [])))

        logger.info(f"OCR complete: {len(all_text)} unique strings, {len(results.get('subtitle_band', []))} subtitle, {len(watermarks)} watermark")

        return {
            "regions": results,
            "all_text": all_text,
            "subtitle": results.get("subtitle_band", []),
            "watermarks": watermarks,
        }

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {
            "error": str(e),
            "regions": {},
            "all_text": [],
            "subtitle": [],
            "watermarks": [],
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
