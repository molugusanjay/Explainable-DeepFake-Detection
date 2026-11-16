import logging
from PIL import Image, ExifTags

# Signatures found ONLY in AI-generation tool outputs (never in real camera metadata).
# NOTE: Generic terms like 'Google', 'Gemini', 'Imagen' are intentionally excluded here
# because Android phones (Motorola, Pixel, Samsung, etc.) embed these words in their
# EXIF Software field and GPS metadata — they would cause massive false positives.
AI_SIGNATURES_BINARY = [
    b'Midjourney',
    b'DALL-E',
    b'Stable Diffusion',
    b'SynthID',
    b'Adobe Firefly',
    b'c2pa',
    b'AI generated',
    b'Generative AI',
]

# Signatures checked ONLY in the EXIF Software tag (more targeted, fewer false positives)
AI_SIGNATURES_SOFTWARE = [
    b'Midjourney',
    b'DALL-E',
    b'Stable Diffusion',
    b'SynthID',
    b'Adobe Firefly',
    b'c2pa',
    b'AI generated',
    b'Generative AI',
    b'Gemini',
    b'Imagen',
]


def analyze_metadata(image_path_or_file):
    """
    Performs a background check on the image metadata to detect AI generation.
    Returns a tuple: (findings_string, has_real_camera).
    - findings_string: human-readable metadata report
    - has_real_camera: True if a camera Make/Model was found in EXIF (strong indicator of a real photo)
    """
    findings = []
    has_real_camera = False

    # 1. Check for AI signatures in raw binary data (catches XMP and IPTC metadata)
    # We use a conservative list here to avoid false-positives from Android/Google metadata
    try:
        if isinstance(image_path_or_file, str):
            with open(image_path_or_file, 'rb') as f:
                data = f.read(500000)  # Read first 500KB
        else:
            data = str(image_path_or_file.info).encode('utf-8', errors='ignore')

        for sig in AI_SIGNATURES_BINARY:
            if sig.lower() in data.lower():
                findings.append(f"⚠️ STRONG AI SIGNATURE DETECTED: {sig.decode('utf-8')}")
    except Exception as e:
        logging.warning(f"Failed to read raw image data for signatures: {e}")

    # 2. Check standard EXIF data (cameras always leave EXIF; AI tools usually strip it)
    try:
        if not isinstance(image_path_or_file, Image.Image):
            img = Image.open(image_path_or_file)
        else:
            img = image_path_or_file

        exif = img.getexif()
        if not exif:
            findings.append(
                "⚠️ NO CAMERA DATA: Standard camera EXIF data is missing. "
                "This is highly common in AI-generated images (or images stripped by "
                "social media/web downloads). Authentic original photos almost always "
                "contain camera metadata."
            )
        else:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ('Make', 'Model', 'LensModel'):
                    has_real_camera = True
                    findings.append(f"Camera Info Found: {tag} = {value}")
                if tag == 'Software':
                    findings.append(f"Software Used: {value}")
                    # Check AI software signatures (more targeted list for Software field)
                    for sig in AI_SIGNATURES_SOFTWARE:
                        if sig.lower() in str(value).encode('utf-8').lower():
                            findings.append(
                                f"⚠️ STRONG AI SIGNATURE DETECTED IN EXIF Software: {sig.decode('utf-8')}"
                            )

            if not has_real_camera:
                findings.append("Standard camera make/model missing from EXIF data.")

    except Exception as e:
        logging.warning(f"Failed to read EXIF data: {e}")

    findings_str = "\n".join(findings) if findings else "No clear metadata indicators found (metadata might be stripped)."
    return findings_str, has_real_camera

