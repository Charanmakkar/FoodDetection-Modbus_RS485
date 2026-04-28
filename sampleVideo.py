import cv2
from ultralytics import YOLO

THRESH_HOLD = 0.5

model = YOLO("best.pt")

# Open Cam1 (usually index = 1)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Step1")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)

    print("Step2")

    detected_objects = []

    print("Step3")

    for r in results:
        print(r)

    print("Step4")

    for r in results:
        if r.probs is None:
            continue

        probs = r.probs.data.cpu().numpy()

        class_id = probs.argmax()
        confidence = probs[class_id]

        print("Confidence:", confidence)

        if confidence < THRESH_HOLD:
            continue

        label = model.names[class_id]
        print("Label:", label)

        detected_objects.append(label)

        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

    cv2.imshow("Live Classification (Cam1)", frame)

    if detected_objects:
        print("Detected objects:", detected_objects)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()