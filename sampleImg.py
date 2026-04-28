import cv2
from ultralytics import YOLO

THRESH_HOLD = 0.5

model = YOLO("best.pt")

image_path = "6.jpg"

frame = cv2.imread(image_path)

print("Step1")

results = model(frame)

print("Step2")

detected_objects = []

print("Step3")

for r in results:
    print(r)

print("Step4")

for r in results:
    # Use probs instead of boxes (classification model)
    if r.probs is None:
        continue

    probs = r.probs.data.cpu().numpy()

    # Get top prediction
    class_id = probs.argmax()
    confidence = probs[class_id]

    print("Confidence:", confidence)

    if confidence < THRESH_HOLD:
        continue

    label = model.names[class_id]
    print("Label:", label)

    detected_objects.append(label)

    # Draw label on image (no box in classification)
    text = f"{label} {confidence:.2f}"
    cv2.putText(frame, text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detected objects:", detected_objects)
