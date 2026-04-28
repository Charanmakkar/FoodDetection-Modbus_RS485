import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("last.pt")


def detect_objects(frame):
    results = model(frame)
    detected_objects = []

    for r in results:
        if r.boxes is None:
            continue  # skip if no detections

        for box in r.boxes:
            confidence = float(box.conf[0])

            # Apply 70% threshold
            if confidence < 0.7:
                continue

            class_id = int(box.cls[0])
            label = model.names[class_id]

            detected_objects.append(label)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show label + confidence
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, detected_objects


def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Camera not working")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = detect_objects(frame)

        cv2.imshow("AI Vision", frame)

        if detected_objects:
            print("Detected objects:", detected_objects)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()