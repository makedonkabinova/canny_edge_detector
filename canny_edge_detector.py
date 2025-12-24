import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return img

def blur(img, type='gaussian'):
    h, w = img.shape
    diag = np.sqrt(h ** 2 + w ** 2)
    if type == 'gaussian':
        s = 0.002 * diag
        kernel_size = int(6 * s + 1)
        if kernel_size % 2 == 0: kernel_size += 1

        # print(kernel_size)
        # print(s)
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), s)
    elif type =='bilateral':
        sigmaS = 0.002 * diag
        sigmaC = 0.1 * np.std(img)
        blurred = cv2.bilateralFilter(img, d=-1, sigmaColor=sigmaC, sigmaSpace=sigmaS)
    return blurred

def apply_sobel(img):
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    convolution_x = cv2.filter2D(src=img, ddepth=-1, kernel=sobel_x)
    convolution_y = cv2.filter2D(src=img, ddepth=-1, kernel=sobel_y)

    return convolution_x, convolution_y

def get_mag_and_directions(gx, gy):
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)
    magnitudes /= magnitudes.max()
    gradient_directions = np.degrees(np.arctan2(gy, gx))  # range of (-180, 180]

    return magnitudes, gradient_directions

def determine_edge_direction(alpha, mag, x, y):
    # 0° (horizontal)
    if (-22.5 <= alpha < 22.5) or (alpha <= -157.5) or (alpha >= 157.5):
        return mag[y, x - 1], mag[y, x + 1]

    # 90° (vertical)
    elif (-112.5 <= alpha < -67.5) or (67.5 <= alpha < 112.5):
        return mag[y - 1, x], mag[y + 1, x]

    # 135° diagonal (NW–SE)
    elif (-157.5 <= alpha < -112.5) or (22.5 <= alpha < 67.5):
        return mag[y - 1, x - 1], mag[y + 1, x + 1]

    # 45° diagonal (NE–SW)
    else:  # (-67.5 ≤ alpha < -22.5) or (112.5 ≤ alpha < 157.5)
        return mag[y + 1, x - 1], mag[y - 1, x + 1]

def non_maximum_suppression(magnitudes, gradient_directions):
    H, W = gradient_directions.shape
    nms = np.zeros_like(magnitudes)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            n1, n2 = determine_edge_direction(gradient_directions[y, x], magnitudes, x, y)
            if (magnitudes[y, x] >= n1) and (magnitudes[y, x] >= n2):
                nms[y, x] = magnitudes[y, x]
    return nms

def otsu(image):
    pixels_total = len(image.flatten())
    background_pixels, foreground_pixels = 0, 0
    hst, _ = np.histogram(image.flatten(), range=(0.0, 1.0), bins=256)
    sum_background = 0
    sum_total = np.sum(np.arange(256) * hst)
    highest_bc_variance = 0.0
    best_threshold = 0
    for threshold in range(256):
        background_pixels += hst[threshold]
        foreground_pixels = pixels_total - background_pixels

        if background_pixels == 0 or foreground_pixels == 0:
            continue
        foreground_w = foreground_pixels / pixels_total
        background_w = background_pixels / pixels_total

        sum_background += threshold * hst[threshold]
        mi_b = sum_background / background_pixels
        mi_f = (sum_total - sum_background) / foreground_pixels

        between_class_variance = background_w * foreground_w
        between_class_variance *= (mi_b - mi_f) ** 2

        if between_class_variance > highest_bc_variance:
            highest_bc_variance = between_class_variance
            best_threshold = threshold
    return best_threshold / 256

def hysteresis_thresholding(nms, th, tl):
    H, W = nms.shape
    strong = nms >= th
    weak = (nms >= tl) & (nms < th)
    output_image = np.zeros_like(nms, dtype=np.uint8)
    output_image[strong] = 1.0
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    stack = list(zip(*np.where(strong)))
    while stack:
        y, x = stack.pop()

        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx

            if 0 <= ny < H and 0 <= nx < W:
                if weak[ny, nx] and output_image[ny, nx] == 0:
                    output_image[ny, nx] = 1
                    stack.append((ny, nx))
    return output_image

def evaluate_edges(edge_img):
    edge_uint8 = (edge_img * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(edge_uint8)
    # print(labels)
    # cv2.imwrite(f"0001_labels_{i}.png", labels.astype(np.uint8))
    sizes = np.bincount(labels.flatten())[1:]  # ignore background
    if len(sizes) == 0:
        return 0

    total_edge_pixels = np.sum(sizes)
    avg_size = np.mean(sizes)
    num_components = len(sizes)

    return (avg_size / num_components) * np.log(total_edge_pixels + 1)


def link_24_connectivity(outputs):
    M = len(outputs)
    h, w = outputs[0].shape

    for m in range(1, M):
        img_n = outputs[m - 1]
        img_n_next = outputs[m]

        visited = np.zeros_like(img_n, dtype=bool)
        ys, xs = np.where(img_n == 1)

        for y, x in zip(ys, xs): ##all pixels have value one in image n
            if visited[y, x]: continue # checking if we visited pixel p(x, y) before (we need previously unselected pixel)
            visited[y, x] = True

            if img_n_next[y, x] == 1: continue #if p+1(x, y) is one, return to previous step

            found_3x3 = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if img_n_next[ny, nx] == 1:
                            found_3x3 = True
                            break
                if found_3x3:
                    break
            if found_3x3:
                continue

            candidates = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if img_n_next[ny, nx] == 1:
                            candidates.append((ny, nx))
            if len(candidates) == 0: continue # around 5x5 neighborhood of image n+1, all pixels are zero value

            for ny, nx in candidates:
                link_shortest_path(img_n_next, nx, ny, x, y)
    return outputs

def link_shortest_path(img_n_next, nx, ny, x, y):
    neighbor_x, neighbor_y = nx, ny
    while (neighbor_x, neighbor_y) != (x, y):
        img_n_next[neighbor_y, neighbor_x] = 1

        if neighbor_x < x: neighbor_x += 1
        elif neighbor_x > x: neighbor_x -= 1

        if neighbor_y < y: neighbor_y += 1
        elif neighbor_y > y: neighbor_y -= 1

    img_n_next[neighbor_y, neighbor_x] = 1

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python canny_edge_detector.py <image_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    image_paths = sorted([
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith(".png")
    ])
    originals = []
    suppressed_images = []
    edges_2d = []
    edges_3d = []

    for img_path in image_paths:

    # input_path = sys.argv[1]
        image = load_image(img_path)
        originals.append(image)

        blurred = blur(image, type='bilateral')
        gx, gy = apply_sobel(blurred)
        magnitudes, directions = get_mag_and_directions(gx, gy)
        suppresed = non_maximum_suppression(magnitudes, directions)

        suppressed_images.append(suppresed)
        # base = img_path.rsplit('.', 1)[0]
        # cv2.imwrite(base + "_suppressed.png", (suppresed * 255).astype(np.uint8))

        nonzero = suppresed[suppresed > 0]
        th = otsu(suppresed)

    #1
    # tl = np.percentile(nonzero, 30)
    # tl = min(tl, th * 0.9)

    #0
    # tl = 0.3 * th

    #2
        best_score = -1
        best_tl = None

        for m in [0.1, 0.2, 0.3, 0.4]:
            tl = m * th
            edges = hysteresis_thresholding(suppresed, th, tl)
            score = evaluate_edges(edges)

            print("threshold: ", tl)
            print("score:", score)
            if score > best_score:
                best_score = score
                best_tl = tl

        tl = best_tl
        print("High threshold: ", th)
        print("Low threshold: ", tl)

        hysteresis_img = hysteresis_thresholding(suppresed, th, tl)
        # cv2.imwrite(base + "_hysteresis.png", (hysteresis_img * 255).astype(np.uint8))
        edges_2d.append(hysteresis_img)

    output_dir = "suppressed_output_bilateral"
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(suppressed_images):
        cv2.imwrite(
            os.path.join(output_dir, f"suppressed_{i+1:03d}.png"),
            (img * 255).astype(np.uint8)
        )

    output_dir = "hysteresis_output_bilateral"
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(edges_2d):
        cv2.imwrite(
            os.path.join(output_dir, f"hysteresis_{i+1:03d}.png"),
            (img * 255).astype(np.uint8)
        )

    edges_3d = link_24_connectivity(edges_2d.copy())

    output_dir = "final_output_bilateral"
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(edges_3d):
        cv2.imwrite(
            os.path.join(output_dir, f"edge_3d_{i+1:03d}.png"),
            (img * 255).astype(np.uint8)
        )
