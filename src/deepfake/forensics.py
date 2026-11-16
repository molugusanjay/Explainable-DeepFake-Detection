import os
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def compute_ela(image, quality=90):
    """
    Computes Error Level Analysis (ELA) to highlight digital manipulations and splicing.
    """
    fd, temp_path = None, None
    try:
        if isinstance(image, str):
            original = Image.open(image).convert('RGB')
        else:
            original = image.copy().convert('RGB')

        # Use a unique temp file so concurrent requests never collide
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        fd = None
        original.save(temp_path, 'JPEG', quality=quality)
        compressed = Image.open(temp_path)

        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        compressed.close()
        return ela_image
    except Exception as e:
        print(f"ELA Error: {e}")
        return image
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def compute_fft(image):
    """
    Computes the 2D Fast Fourier Transform (FFT) magnitude spectrum.
    Highlights repeating frequency artifacts common in GANs and DeepFakes.
    """
    try:
        if isinstance(image, str):
            img = Image.open(image).convert('L')
        else:
            img = image.copy().convert('L')
            
        img_array = np.array(img)
        f = np.fft.fft2(img_array)
        fshift = np.fft.fftshift(f)
        
        # Add 1 to avoid log(0)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Normalize to 0-255
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        
        # Apply a colormap for better visibility (optional, but looks highly professional)
        colored_spectrum = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)
        # Convert BGR to RGB
        colored_spectrum = cv2.cvtColor(colored_spectrum, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(colored_spectrum)
    except Exception as e:
        print(f"FFT Error: {e}")
        return image
