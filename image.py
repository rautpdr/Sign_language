import math
import os
import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
save_flag = False


# ✅ Get correct absolute folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(BASE_DIR, "Data", "C")

# ✅ Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # detects hands and draws landmarks

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Make sure coordinates stay within the frame
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        # Crop and prepare white background
        imgCrop = img[y1:y2, x1:x2]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        # Resize and paste into white image
        if aspectRatio > 1:  # height > width
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:  # width >= height
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Save image if save_flag is active
        if save_flag:
            counter += 1
            filename = os.path.join(folder, f"Image_{time.time()}.jpg")
            cv2.imwrite(filename, imgWhite)
            print(f"[Saved] {filename}")

        # Display cropped and white canvas
        #cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Original", img)
    # Save on keypress "s"
    # Key handler
    key = cv2.waitKey(1)
    if key == ord('s'):
        save_flag = True
        #print("[INFO] Saving started... Press 'q' to stop.")
        counter += 1
        print(counter)
    elif key == ord('q'):
        save_flag = False
        print("[INFO] Saving stopped.")


