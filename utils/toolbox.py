import os
import base64
import time

def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text

def block_timer(name=""):
    class TimerContextManager:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.end = time.perf_counter()
            print(f"{self.name} Time Cost: {(self.end - self.start) / 3600:.4f} hours")
            
    return TimerContextManager(name)

def encode_image_to_base64(image_path):
    """Read image file and encode to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_mime_type(image_path):
    """Determine MIME type based on file extension"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(ext, 'image/jpeg')