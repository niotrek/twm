from src.detector import Detector
from src.letter_classifier import LetterClassifier
from src.process_image import ImageProcessor
import random
import os


if __name__ == "__main__":
    detector = Detector()
    img_processor = ImageProcessor()
    letter_classifier = LetterClassifier()
    detector.load_models()
    img_name = random.choice(os.listdir(detector.testset_path))
    img_path = os.path.join(detector.testset_path, img_name)
    img = detector._load_image(img_path)
    plates = detector.find_plates(img)
    for plate in plates:
        cropped = img_processor.process_image(plate, visualize=True)
        for crop in cropped:
            letter_classifier.predict_image(crop)
    

    

