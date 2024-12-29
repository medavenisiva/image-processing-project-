import cv2
import numpy as np
from tkinter import Tk, Button, Canvas, filedialog, Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

original_image = None
processed_image = None

def open_image():
    
    global original_image, processed_image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        original_image = cv2.imread(file_path)
        processed_image = original_image.copy()
        display_image(original_image)

def save_image():
    
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, processed_image)

def display_image(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, anchor="nw", image=image)

def rgb_to_gray():
    global processed_image
    if original_image is None:
        return
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)
def sharpen_image():
    global processed_image
    if original_image is None:
        return
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    processed_image = cv2.filter2D(original_image, -1, kernel)
    display_image(processed_image)

def canny_edge_detection():
    global processed_image
    if original_image is None:
        return
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)

def erosion():
    global processed_image
    if original_image is None:
        return
    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.erode(original_image, kernel, iterations=1)
    display_image(processed_image)

def dilation():
    global processed_image
    if original_image is None:
        return
    kernel = np.ones((5, 5), np.uint8)
    processed_image = cv2.dilate(original_image, kernel, iterations=1)
    display_image(processed_image)

def apply_sepia():
    global processed_image
    if original_image is None:
        return
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
    processed_image = cv2.transform(original_image, sepia_filter)
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    display_image(processed_image)

def adjust_brightness_contrast(brightness=30, contrast=50):
    global processed_image
    if original_image is None:
        return
    processed_image = cv2.convertScaleAbs(original_image, alpha=contrast / 50, beta=brightness - 50)
    display_image(processed_image)


def rgb_to_binary():
    global processed_image
    if original_image is None:
        return
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)
def gray_to_binary():
    global processed_image
    if original_image is None:
        return
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)

def apply_transformation(transformation):
    global processed_image
    if original_image is None:
        return

    if transformation == "":
        processed_image = np.sqrt(original_image).astype(np.uint8)
    elif transformation == "Powers":
        processed_image = np.power(original_image, 2).clip(0, 255).astype(np.uint8)
    elif transformation == "Negative":
        processed_image = 255 - original_image
    elif transformation == "Log":
        c = 255 / np.log(1 + np.max(original_image))
        processed_image = (c * np.log(1 + original_image)).astype(np.uint8)
    elif transformation == "Inverse Log":
        c = 255 / np.log(256)
        processed_image = (np.exp(original_image / c) - 1).clip(0, 255).astype(np.uint8)
    elif transformation == "Gamma":
        gamma = 2.2
        inv_gamma = 1 / gamma
        processed_image = np.power(original_image / 255.0, inv_gamma) * 255
        processed_image = processed_image.astype(np.uint8)
    elif transformation == "Blurring":
        processed_image = cv2.GaussianBlur(original_image, (15, 15), 0)
    display_image(processed_image)

def plot_histogram(operation):
    if original_image is None:
        return

    if operation == "Gray Histogram":
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        plt.hist(gray.ravel(), 256, [0, 256])
        plt.title("Gray Histogram")
    elif operation == "RGB Histogram":
        for i, col in enumerate(['b', 'g', 'r']):
            plt.hist(original_image[:, :, i].ravel(), 256, [0, 256], color=col)
        plt.title("RGB Histogram")
    plt.show()

def histogram_equalization():
    global processed_image
    if original_image is None:
        return
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)

def edge_detection(direction):
    global processed_image
    if original_image is None:
        return

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    kernel = None
    if direction == "Horizontal":
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif direction == "Vertical":
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif direction == "Diagonal Left":
        kernel = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    elif direction == "Diagonal Right":
        kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])

    edges = cv2.filter2D(gray, -1, kernel)
    processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)

def add_noise(noise_type):
    global processed_image
    if original_image is None:
        return

    if noise_type == "Salt & Pepper":
        noise = np.random.choice((0, 255), original_image.shape, p=[0.99, 0.01]).astype(np.uint8)
        processed_image = cv2.add(original_image, noise)
    elif noise_type == "Gaussian":
        mean = 0
        stddev = 25
        noise = np.random.normal(mean, stddev, original_image.shape).astype(np.uint8)
        processed_image = cv2.add(original_image, noise)

    display_image(processed_image)
def apply_watermark():
    global processed_image
    if original_image is None:
        return

    watermark = np.zeros_like(original_image)
    cv2.putText(watermark, "shiva", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    alpha = 0.5
    processed_image = cv2.addWeighted(original_image, 1 - alpha, watermark, alpha, 0)
    display_image(processed_image)
def morphological_operation(operation="Opening"):
    global processed_image
    if original_image is None:
        return

    kernel = np.ones((5, 5), np.uint8)
    if operation == "Opening":
        processed_image = cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        processed_image = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)
    elif operation == "Gradient":
        processed_image = cv2.morphologyEx(original_image, cv2.MORPH_GRADIENT, kernel)
    display_image(processed_image)
def restore_photo():
    global processed_image
    if original_image is None:
        return

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    processed_image = cv2.inpaint(original_image, mask, 7, cv2.INPAINT_TELEA)
    display_image(processed_image)
def pencil_sketch():
    global processed_image
    if original_image is None:
        return

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    processed_image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)

def clahe_equalization():
    global processed_image
    if original_image is None:
        return

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    display_image(processed_image)
def watershed_segmentation():
    global processed_image
    if original_image is None:
        return

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original_image, markers)
    processed_image = original_image.copy()
    processed_image[markers == -1] = [255, 0, 0]
    display_image(processed_image)

def kmeans_segmentation():
    global processed_image
    if original_image is None:
        return

    # Convert the image to a 2D array of pixels
    Z = original_image.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 4  # Number of clusters
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    processed_image = segmented_image.reshape(original_image.shape)
    display_image(processed_image)



imag = Tk()
imag.title("image processing tools")
imag.geometry("900x900")

canvas = Canvas(imag, width=500, height=256, bg="gray")
canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

Button(imag, text="Open Image", command=open_image).grid(row=1, column=0, padx=10, pady=5)
Button(imag, text="Save Image", command=save_image).grid(row=1, column=1, padx=10, pady=5)

Button(imag, text="RGB to Gray", command=rgb_to_gray).grid(row=2, column=0, padx=10, pady=5)
Button(imag, text="RGB to Binary", command=rgb_to_binary).grid(row=2, column=1, padx=10, pady=5)

Button(imag, text="Negative", command=lambda: apply_transformation("Negative")).grid(row=3, column=0, padx=10, pady=5)
Button(imag, text="Gamma", command=lambda: apply_transformation("Gamma")).grid(row=3, column=1, padx=10, pady=5)

Button(imag, text="Gray Histogram", command=lambda: plot_histogram("Gray Histogram")).grid(row=4, column=0, padx=10, pady=5)
Button(imag, text="RGB Histogram", command=lambda: plot_histogram("RGB Histogram")).grid(row=4, column=1, padx=10, pady=5)
Button(imag, text="K-Means seg", command=kmeans_segmentation).grid(row=5, column=1, padx=10, pady=5)
Button(imag, text="Histogram Equalization", command=histogram_equalization).grid(row=4, column=2, padx=10, pady=5)
Button(imag, text="Add Salt & Pepper Noise", command=lambda: add_noise("Salt & Pepper")).grid(row=5, column=0, padx=10, pady=5)
Button(imag, text="Sharpen", command=sharpen_image).grid(row=1, column=2, padx=10, pady=5)
Button(imag, text="Canny Edge", command=canny_edge_detection).grid(row=2, column=2, padx=10, pady=5)
Button(imag, text="Erosion", command=erosion).grid(row=3, column=2, padx=10, pady=5)
Button(imag, text="Dilation", command=dilation).grid(row=1, column=3, padx=10, pady=5)
Button(imag, text="Sepia", command=apply_sepia).grid(row=2, column=3, padx=10, pady=5)
Button(imag, text="Apply Watermark", command=apply_watermark).grid(row=3, column=3, padx=10, pady=5)
Button(imag, text="Morphological Op", command=lambda: morphological_operation("Opening")).grid(row=4, column=3, padx=10, pady=5)
Button(imag, text="Restore Photo", command=restore_photo).grid(row=5, column=3, padx=3, pady=5)
Button(imag, text="Pencil Sketch", command=pencil_sketch).grid(row=6, column=3, padx=10, pady=5)
Button(imag, text="Watershed Segmentation", command=watershed_segmentation).grid(row=5, column=2, padx=10, pady=5)
Button(imag, text="CLAHE", command=clahe_equalization).grid(row=6, column=2, padx=10, pady=5)

Button(imag, text="Brightness/Contrast", command=lambda: adjust_brightness_contrast(80, 70)).grid(row=7, column=1, padx=10, pady=5)
imag.mainloop()


