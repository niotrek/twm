from src.detector import Detector
from src.letter_classifier import LetterClassifier
from src.process_image import ImageProcessor
import random
import os


if __name__ == "__main__":
    detector = Detector()
    img_processor = ImageProcessor()
    letter_classifier = LetterClassifier('./data_classificator')
    detector.load_models()
    letter_classifier.load_best_model('./Best_model/LetterClasifier/best_model_2024-06-08_13-11-41.keras')
    # img_name = random.choice(os.listdir(detector.testset_path))
    # img_path = os.path.join(detector.testset_path, img_name)
    # img = detector._load_image(img_path)
    images_names = os.listdir(detector.testset_path)
    for img_name in images_names:
        img_path = os.path.join(detector.testset_path, img_name)
        img = detector._load_image(img_path)
        plates = detector.find_plates(img)
        for plate in plates:
            cropped = img_processor.process_image(plate, visualize=True)
            for crop in cropped:
                letter_classifier.predict_image(crop)
        print(f"Image {img_name} processed.")
        

    

