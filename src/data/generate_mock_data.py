import os
from src.config import CHEST_DIR, BRAIN_DIR, CHEST_CLASSES, BRAIN_CLASSES

def generate_mock_images(directory, classes, num_images_per_class=10):
    """
    Generates minimal valid JPEG test images for pipeline validation
    when raw medical datasets have not yet been downloaded.
    """
    for cls in classes:
        cls_dir = os.path.join(directory, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(num_images_per_class):
            with open(os.path.join(cls_dir, f'mock_{i}.jpg'), 'wb') as f:
                # Valid 1x1 black JPEG header & data
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x00\xff\xd9')
    print(f"Generated {num_images_per_class} mock valid JPEG images for each class in {directory}")

if __name__ == "__main__":
    generate_mock_images(CHEST_DIR, CHEST_CLASSES, 10)
    generate_mock_images(BRAIN_DIR, BRAIN_CLASSES, 10)
