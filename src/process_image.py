import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class ImageProcessor:
    def process_image(self, image=None, image_path=None, save_crops=False, visualize=True):
        if image is None and image_path is None:
            return None
        elif isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image

        # Wczytaj obraz
        image = cv2.resize(image, (200, 100))

        # Konwersja obrazu na skalę szarości
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Zastosowanie filtru Otsu do progowania
        _, otsu_mask = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Znajdź składowe połączone
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            otsu_mask, connectivity=8
        )

        # Znajdź składową połączoną z największym polem
        areas = stats[1:, cv2.CC_STAT_AREA]  # Pomiń pierwszy element, który jest tłem
        largest_component_index = np.argmax(areas) + 1  # Indeks największej składowej
        x, y, w, h = (
            stats[largest_component_index, cv2.CC_STAT_LEFT],
            stats[largest_component_index, cv2.CC_STAT_TOP],
            stats[largest_component_index, cv2.CC_STAT_WIDTH],
            stats[largest_component_index, cv2.CC_STAT_HEIGHT],
        )

        # Wytnij i odwróć obraz największej składowej
        otsu_mask = otsu_mask[y : y + h, x : x + w]
        otsu_mask_reversed = cv2.bitwise_not(otsu_mask)

        # Znajdź składowe połączone
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            otsu_mask_reversed, connectivity=8
        )
        
        areas = stats[0:, cv2.CC_STAT_AREA]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        
        lower_bound = mean_area*0.1 #- std_area #* 0.25
        upper_bound = mean_area*0.95 #+ std_area #* 0.5

        # Acceptable aspect ratio range
        min_aspect_ratio = 1.5
        max_aspect_ratio = 200
        
        # Acceptable centroid y-coordinate range
        min_y_coord = 15
        max_y_coord = 90
        
        # Acceptable centroid y-coordinate range
        min_x_coord = 5
        
        valid_area=[]
        for i in range(1, num_labels):
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            aspect_ratio = h / w
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                if min_y_coord <= centroids[i][1] <= max_y_coord and  min_x_coord<=centroids[i][0]:
                    valid_area.append(stats[0:, cv2.CC_STAT_AREA])
        mean_area = np.mean(valid_area)
        std_area = np.std(valid_area)         

        # Znajdź i posortuj bounding boxy składowych połączonych według współrzędnej x
        filtered_components = []
        filtered_centroids = []
        for i in range(1, num_labels):
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
                        
            # print("lb",lower_bound)
            # print("area",stats[i, cv2.CC_STAT_AREA])
            # print("ub",upper_bound)
            # print("mean",mean_area)
            # print("std",std_area)
            # print()
            # # print(x, y, w, h)
            if lower_bound <= stats[i, cv2.CC_STAT_AREA] <= upper_bound:
                aspect_ratio = h / w
                if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                    if (
                        centroids[i][0] >= x
                        and centroids[i][0] <= x + w
                        and centroids[i][1] >= y
                        and centroids[i][1] <= y + h
                    ):
                        if min_y_coord <= centroids[i][1] <= max_y_coord and  min_x_coord<=centroids[i][0]:
                            
                            # print(centroids[i])

                            filtered_components.append((x, y, w, h))
                            filtered_centroids.append(centroids[i])

        filtered_components.sort()  # Sortuj według współrzędnej x

        # Wytnij i odwróć obrazy z maski
        cropped_masks = []
        for i, (x, y, w, h) in enumerate(filtered_components):
            cropped_mask = otsu_mask_reversed[y : y + h, x : x + w]
            cropped_mask_reversed = cv2.bitwise_not(cropped_mask)
            padding = 5  # Długość boku białego obszaru wokół wyciętego fragmentu
            padded_mask = np.zeros((h + 2 * padding, w + 2 * padding), dtype=np.uint8)
            padded_mask[padding : padding + h, padding : padding + w] = cropped_mask
            padded_mask = cv2.bitwise_not(padded_mask)
            cropped_masks.append(padded_mask)

            # Zapisz każdy wycięty obraz, jeśli opcja zapisu jest ustawiona na True
            if save_crops:
                if not os.path.exists("better_data/crop"):
                    os.makedirs("better_data/crop")

                if not os.path.exists(
                    "better_data/crop/" + image_path.split("/")[-1].split(".")[0]
                ):
                    os.makedirs(
                        "better_data/crop/" + image_path.split("/")[-1].split(".")[0]
                    )

                filename = (
                    "better_data/crop/"
                    + image_path.split("/")[-1].split(".")[0]
                    + f"/crop_{i+1}.png"
                )
                # print(filename)
                cv2.imwrite(filename, padded_mask)

        # centroids = centroids[1:]
        filtered_centroids = np.array(filtered_centroids)
        y_coordinates = centroids[:, 1].reshape(-1, 1)

        dbscan = DBSCAN(
            eps=10, min_samples=8
        )  # Parametry eps (odległość) i min_samples (minimalna liczba punktów w klastrze)
        cluster_labels = dbscan.fit_predict(y_coordinates)

        clustered_image = np.zeros_like(gray_image)

        for i, (_, y) in enumerate(filtered_centroids):
            cluster_idx = cluster_labels[i]
            cv2.circle(
                clustered_image,
                (int(filtered_centroids[i][0]), int(y)),
                int(cluster_idx) + 4,
                int(cluster_idx) + 1,
                -1,
            )  # +1, aby uniknąć koloru tła

        # Opcjonalnie: Pokaż wyniki
        if visualize:
            # print(image_path)
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 5, 1)
            plt.title("Oryginalny Obraz")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 5, 2)
            plt.title("Obraz w Skali Szarości")
            plt.imshow(gray_image, cmap="gray")

            plt.subplot(1, 5, 3)
            plt.title("Maska Otsu")
            plt.imshow(otsu_mask, cmap="gray")

            plt.subplot(1, 5, 4)
            plt.title("Odwrócona Maska Otsu")
            plt.imshow(otsu_mask_reversed, cmap="gray")

            # plt.subplot(1, 5, 5)
            # plt.imshow(clustered_image, cmap="jet")  # Użyj mapy kolorów 'jet'
            # plt.title("Wyniki Klastrów")

            plt.show()

            # Wyświetl wycięte obrazy z odwróconej maski
            plt.figure(figsize=(20, 5))
            for i, cropped_mask in enumerate(cropped_masks):
                plt.subplot(1, len(cropped_masks), i + 1)
                plt.imshow(cropped_mask, cmap="gray")
                plt.title(f"Obszar {i+1}")
                plt.axis("off")

            plt.show()

        return cropped_masks


if __name__ == "__main__":
    processor = ImageProcessor()
    image = cv2.imread('better_data/plate/image_00.png')
    cropped = processor.process_image(
        image=image,
        save_crops=False,
        visualize=True,
    )