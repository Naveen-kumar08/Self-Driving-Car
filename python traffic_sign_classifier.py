import cv2
import numpy as np
import os
import csv
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/traffic_sign_model.h5")

# Load sign names
signnames = {}
with open("data/signnames.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        signnames[int(row[0])] = row[1]

# Create output folder
os.makedirs("output", exist_ok=True)

for image_name in os.listdir("webimages"):

    path = os.path.join("webimages", image_name)
    img = cv2.imread(path)

    if img is None:
        continue

    print("\nProcessing:", image_name)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect RED traffic signs
    lower_red1 = np.array([0,70,50])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([170,70,50])
    upper_red2 = np.array([180,255,255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask_red1 + mask_red2

    # Detect BLUE traffic signs
    lower_blue = np.array([90,60,60])
    upper_blue = np.array([130,255,255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine masks
    mask = red_mask + blue_mask

    # Clean noise
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No sign detected")
        continue

    # Largest contour = likely traffic sign
    cnt = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(cnt)

    # Extract ROI
    roi = img[y:y+h, x:x+w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Accuracy improvement
    gray = cv2.equalizeHist(gray)

    roi = cv2.resize(gray,(32,32))

    roi = roi/255.0
    roi = roi.reshape(1,32,32,1)

    prediction = model.predict(roi, verbose=0)

    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    class_name = signnames.get(class_id, "Unknown")

    print("Detected:", class_name)
    print("Confidence:", round(confidence*100,2), "%")

    # Tight bounding box
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(img,label,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(0,255,0),2)

    # Save result
    output_path = os.path.join("output",image_name)
    cv2.imwrite(output_path,img)

    cv2.imshow("Traffic Sign Detection",img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
