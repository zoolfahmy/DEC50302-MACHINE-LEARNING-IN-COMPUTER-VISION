import cv2 
import os

# Example of a robust approach
image_path = r'D:/Machine Learning Course/SA github/chapter2/images/park.jpg' 
# Replace with your actual path

if not os.path.exists(image_path):
    print(f"Error: Image file not found at '{image_path}'") 
else:
    img = cv2.imread(image_path)

    if img is None:
        print("Error: The image cannot be read. Check file permissions.") 
    else:
        # Proceed with image Blurring Technique
        box = cv2.blur(img, (7, 7))
        gauss = cv2.GaussianBlur(img, (7, 7), 0)
        median = cv2.medianBlur(img, 7)
        bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        cv2.imwrite("blur_box.png", box)
        cv2.imshow("Box blur", box)
        cv2.imwrite("blur_gaussian.png", gauss)
        cv2.imshow("Gaussian blur", gauss)
        cv2.imwrite("blur_median.png", median)
        cv2.imshow("Median blur", median)
        cv2.imwrite("blur_bilateral.png", bilateral)
        cv2.imshow("Bilateral blur", bilateral)
        cv2.waitKey(0)  # keep window open until a key press    
        cv2.destroyAllWindows()
        