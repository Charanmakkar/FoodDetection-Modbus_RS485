import cv2
import json
import time
from pymodbus.client import ModbusSerialClient
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
JSON_MAP_FILE = "item_temps.json"

RS485_PORT = 'COM5'
BAUD_RATE = 9600
PID_SLAVE_ID = 1

READ_LOCATIONS = [0x1000, 0x1001, 0x1012]
WRITE_LOCATIONS = [0x1001]

MODEL_PT = "best.pt"

CONF_THRESHOLD = 0.5
FRAME_SKIP = 5   # run YOLO every N frames
WRITE_DELAY = 2  # seconds debounce

# ==========================================
# 2. INIT
# ==========================================
# Load JSON
try:
    with open(JSON_MAP_FILE, 'r') as f:
        item_temp_map = json.load(f)
except:
    print("JSON not found, creating dummy...")
    item_temp_map = {"TargetObject": [150, 160, 170]}
    with open(JSON_MAP_FILE, 'w') as f:
        json.dump(item_temp_map, f, indent=4)

# Modbus init
modbus_client = ModbusSerialClient(
    port=RS485_PORT,
    baudrate=BAUD_RATE,
    timeout=0.1,
    stopbits=1,
    bytesize=8,
    parity='N'
)

if not modbus_client.connect():
    print("❌ Modbus connection failed")
else:
    print("✅ Modbus connected")

# YOLO init
print("Loading model...")
model = YOLO(MODEL_PT)
print("Model loaded. Classes:", model.names)

# Camera init
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not working")
    exit()

# ==========================================
# 3. FUNCTIONS
# ==========================================
def read_pid_data():
    try:
        result = modbus_client.read_holding_registers(
            address=READ_LOCATIONS[0],
            count=1,
            device_id=PID_SLAVE_ID
        )
        if not result.isError():
            return result.registers, True
    except Exception as e:
        print("Read error:", e)

    return ["ERR"] * len(READ_LOCATIONS), False


def write_pid_data(temp_values):
    try:
        modbus_client.write_registers(
            address=WRITE_LOCATIONS[0],
            values=temp_values*10,
            device_id=PID_SLAVE_ID
        )
        print("✅ Written to PID:", temp_values)
    except Exception as e:
        print("Write error:", e)

# ==========================================
# 4. MAIN LOOP
# ==========================================
frame_count = 0
last_detected_item = None
last_write_time = 0

current_temperatures = ["-", "-", "-"]
comm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame error")
        break

    frame_count += 1
    detected_item_this_frame = None

    # --------------------------------------
    # A. YOLO DETECTION (throttled)
    # --------------------------------------
    if frame_count % FRAME_SKIP == 0:
        results = model(frame, verbose=False)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])

                if conf > CONF_THRESHOLD:
                    class_id = int(box.cls[0])
                    detected_item_this_frame = model.names[class_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame,
                                f"{detected_item_this_frame} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2)

                    print(f"Detected: {detected_item_this_frame} ({conf:.2f})")
                    break

    # --------------------------------------
    # B. MODBUS WRITE LOGIC
    # --------------------------------------
    if detected_item_this_frame:
        if detected_item_this_frame != last_detected_item:
            if time.time() - last_write_time > WRITE_DELAY:

                if detected_item_this_frame in item_temp_map:
                    temps = item_temp_map[detected_item_this_frame]
                    write_pid_data(temps)
                else:
                    print("⚠ Not in JSON:", detected_item_this_frame)

                last_detected_item = detected_item_this_frame
                last_write_time = time.time()

    # --------------------------------------
    # C. MODBUS READ
    # --------------------------------------
    current_temperatures, comm_active = read_pid_data()
    current_temperatures[0] = current_temperatures[0]/10
    # --------------------------------------
    # D. UI
    # --------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (360, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    status_color = (0, 255, 0) if comm_active else (0, 0, 255)
    cv2.circle(frame, (30, 30), 8, status_color, -1)
    cv2.putText(frame, "RS485", (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Last: {last_detected_item or 'None'}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"T1: {current_temperatures[0]}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

##    cv2.putText(frame, f"T2: {current_temperatures[1]}",
##                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
##
##    cv2.putText(frame, f"T3: {current_temperatures[2]}",
##                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Vision + PID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
modbus_client.close()
