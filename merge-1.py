import cv2
import json
import time
from pymodbus.client import ModbusSerialClient
from ultralytics import YOLO

# ==========================================
# CONFIG
# ==========================================
THRESH_HOLD = 0.5
MODEL_PT = "best.pt"

RS485_PORT = 'COM9'
BAUD_RATE = 9600
SLAVE_ID = 1

WRITE_REGISTER = 0x1001
READ_REGISTER = 0x1000

WRITE_DELAY = 2  # seconds

JSON_MAP_FILE = "item_temps.json"

# ==========================================
# INIT
# ==========================================
# Load mapping
try:
    with open(JSON_MAP_FILE, 'r') as f:
        item_temp_map = json.load(f)
except:
    item_temp_map = {"hamburger": [150]}
    with open(JSON_MAP_FILE, 'w') as f:
        json.dump(item_temp_map, f, indent=4)

# Modbus
modbus_client = ModbusSerialClient(
    port=RS485_PORT,
    baudrate=BAUD_RATE,
    stopbits=1,
    bytesize=8,
    parity='N',
    timeout=0.5
)

if modbus_client.connect():
    print("✅ Modbus Connected")
else:
    print("❌ Modbus Failed")

# YOLO
model = YOLO(MODEL_PT)

# Camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera error")
    exit()

# ==========================================
# FUNCTIONS
# ==========================================
def read_pid():
    try:
        res = modbus_client.read_holding_registers(
            address=READ_REGISTER,
            count=1,
            device_id=SLAVE_ID
        )
        if not res.isError():
            return res.registers[0] / 10, True
    except Exception as e:
        print("Read error:", e)

    return "-", False


def write_pid(temp):
    try:
        modbus_client.write_register(
            address=WRITE_REGISTER,
            value=int(temp * 10),
            device_id=SLAVE_ID
        )
        print("✅ Written:", temp)
    except Exception as e:
        print("Write error:", e)


# ==========================================
# MAIN LOOP
# ==========================================
last_label = None
last_write_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    detected_label = None
    confidence = 0

    # --------------------------------------
    # CLASSIFICATION LOGIC
    # --------------------------------------
    for r in results:
        if r.probs is None:
            continue

        probs = r.probs.data.cpu().numpy()
        class_id = probs.argmax()
        confidence = probs[class_id]

        if confidence >= THRESH_HOLD:
            detected_label = model.names[class_id]

            # Draw label
            text = f"{detected_label} {confidence:.2f}"
            cv2.putText(frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    # --------------------------------------
    # MODBUS WRITE LOGIC
    # --------------------------------------
    if detected_label:
        if detected_label != last_label:
            if time.time() - last_write_time > WRITE_DELAY:

                if detected_label in item_temp_map:
                    temp_val = item_temp_map[detected_label][0]
                    write_pid(temp_val)
                else:
                    print("⚠ Not mapped:", detected_label)

                last_label = detected_label
                last_write_time = time.time()

    # --------------------------------------
    # MODBUS READ
    # --------------------------------------
    temp, comm_ok = read_pid()

    # --------------------------------------
    # UI
    # --------------------------------------
    color = (0, 255, 0) if comm_ok else (0, 0, 255)

    cv2.circle(frame, (30, 30), 8, color, -1)
    cv2.putText(frame, "RS485", (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    cv2.putText(frame, f"Last: {last_label or 'None'}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)

    cv2.putText(frame, f"T1: {temp}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)

    cv2.imshow("Vision + Modbus", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
modbus_client.close()