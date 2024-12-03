import pandas as pd
import cv2
import os
import numpy as np

def process_single_file(image_path, csv_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    df = pd.read_csv(csv_path)

    def get_center_depth(row):
        x = int(row['centroid_x'])
        y = int(row['centroid_y'])
        return image[y, x]

    df['center_depth'] = df.apply(get_center_depth, axis=1)

    def calculate_polygon_stats(row):
        polygon = eval(row['polygon'])
        mask = np.zeros(image.shape, dtype=np.uint8)
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        pixel_values = masked_image[mask == 1]

        mean_val = np.mean(pixel_values)
        std_val = np.std(pixel_values)
        var_val = np.var(pixel_values)

        return pd.Series([mean_val, std_val, var_val], index=['mean_gray', 'std_gray', 'var_gray'])

    df[['mean_gray', 'std_gray', 'var_gray']] = df.apply(calculate_polygon_stats, axis=1)

    def calculate_symmetry(contour):
        contour = contour.squeeze()
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return 0
        x_center = int(moments['m10'] / moments['m00'])
        y_center = int(moments['m01'] / moments['m00'])

        left_side = contour[contour[:, 0] < x_center]
        right_side = contour[contour[:, 0] > x_center]
        right_side[:, 0] = 2 * x_center - right_side[:, 0]

        if len(left_side) == 0 or len(right_side) == 0:
            return 0

        distance = np.linalg.norm(left_side[:, None] - right_side[None, :], axis=2)
        min_distance = np.min(distance, axis=1)
        symmetry_score = 1 - np.mean(min_distance) / np.linalg.norm(contour.max(axis=0) - contour.min(axis=0))

        return symmetry_score

    def calculate_polygon_features(row):
        polygon = eval(row['polygon'])
        polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

        area = cv2.contourArea(polygon)
        perimeter = cv2.arcLength(polygon, True)
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = rect[1][0]
        height = rect[1][1]

        elongation = max(width, height) / min(width, height) if min(width, height) != 0 else 0

        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

        rectangularity = area / (width * height) if width != 0 and height != 0 else 0

        aspect_ratio = width / height if height != 0 else 0

        hull = cv2.convexHull(polygon)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area != 0 else 0

        if len(polygon) >= 5: 
            ellipse = cv2.fitEllipse(polygon)
            (a, b) = (ellipse[1][0] / 2, ellipse[1][1] / 2)  
            eccentricity = np.sqrt(1 - (min(a, b) / max(a, b))**2)
        else:
            eccentricity = 0

        symmetry = calculate_symmetry(polygon)

        return pd.Series([elongation, circularity, rectangularity, aspect_ratio, convexity, eccentricity, symmetry],
                         index=['elongation', 'circularity', 'rectangularity', 'aspect_ratio', 'convexity', 'eccentricity', 'symmetry'])

    df[['elongation', 'circularity', 'rectangularity', 'aspect_ratio', 'convexity', 'eccentricity', 'symmetry']] = df.apply(calculate_polygon_features, axis=1)

    output_csv_path = os.path.join(output_folder, os.path.basename(csv_path))

    df.to_csv(output_csv_path, index=False)

    print(f"新CSV文件已保存到 {output_csv_path}")

image_path = '513d6902fdc9f03587004592.jpg' 
csv_path = '513d6902fdc9f03587004592.csv' 
output_folder = 'nodelist'  

os.makedirs(output_folder, exist_ok=True)

process_single_file(image_path, csv_path, output_folder)