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

WRITE_REGISTER_setPoint = 0x1001
WRITE_REGISTER_alarm01LowValue = 0x1002
WRITE_REGISTER_alarm01HighValue = 0x1003
WRITE_REGISTER_alarm02LowValue = 0x1002
WRITE_REGISTER_alarm02HighValue = 0x1003

READ_REGISTER_currentPoint = 0x1000
READ_REGISTER_setPoint = 0x1001


WRITE_DELAY = 2  # seconds

JSON_MAP_FILE = "food_temp_data.json"

# ==========================================
# INIT
# ==========================================

# ---- Load JSON and create name → data map ----
try:
    with open(JSON_MAP_FILE, 'r') as f:
        raw_data = json.load(f)

    item_temp_map = {}
    for key, value in raw_data.items():
        name = value["name"]
        item_temp_map[name] = value

    print("✅ JSON Loaded Successfully")

except Exception as e:
    print("❌ JSON Load Error:", e)
    item_temp_map = {}

# ---- Modbus Init ----
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

# ---- YOLO Init ----
model = YOLO(MODEL_PT)

# ---- Camera Init ----
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera Error")
    exit()

# ==========================================
# FUNCTIONS
# ==========================================
def read_SET_POINT_TEMP():
    try:
        res = modbus_client.read_holding_registers(
            address=READ_REGISTER_setPoint,
            count=1,
            device_id=SLAVE_ID
        )
        if not res.isError():
            return res.registers[0] / 10, True
    except Exception as e:
        print("Read error:", e)

    return "-", False

def read_CURRENT_POINT_TEMP():
    try:
        res = modbus_client.read_holding_registers(
            address=READ_REGISTER_currentPoint,
            count=1,
            device_id=SLAVE_ID
        )
        if not res.isError():
            return res.registers[0] / 10, True
    except Exception as e:
        print("Read error:", e)

    return "-", False


def write_SET_POINT_TEMP(temp):
    try:
        modbus_client.write_register(
            address=WRITE_REGISTER_setPoint,
            value=int(temp * 10),  # scale
            device_id=SLAVE_ID
        )
        print(f"✅ Written to PID: {temp} °C")
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
        print("Frame error")
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

            text = f"{detected_label} {confidence:.2f}"
            cv2.putText(frame, text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

    # --------------------------------------
    # MODBUS WRITE LOGIC
    # --------------------------------------
    if detected_label:
        if (detected_label != last_label) or (time.time() - last_write_time > WRITE_DELAY):

            if detected_label in item_temp_map:
                temp_val = item_temp_map[detected_label]["temp_maintain"]
                write_SET_POINT_TEMP(temp_val)
            else:
                print("⚠ Not mapped:", detected_label)

            last_label = detected_label
            last_write_time = time.time()

    # --------------------------------------
    # MODBUS READ
    # --------------------------------------
    currentPointTemp, currentPoint_comm_ok = read_CURRENT_POINT_TEMP()
    setPointTemp, setPoint_comm_ok = read_SET_POINT_TEMP()

    # --------------------------------------
    # UI
    # --------------------------------------
    color = (0, 255, 0) if currentPoint_comm_ok else (0, 0, 255)

    cv2.circle(frame, (30, 30), 8, color, -1)
    cv2.putText(frame, "RS485", (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    
    # TEMP TEXT ON LEFT SIDE 
    cv2.putText(frame, f"Last: {last_label or 'None'}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)

    cv2.putText(frame, f"CurrentTemp: {currentPointTemp}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    
    cv2.putText(frame, f"SetTemp: {setPointTemp}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    
    # LOWER RIGHT SIDE TEXT
    if detected_label and detected_label in item_temp_map:
        item = item_temp_map[detected_label]

        # cv2.putText(frame, f"NAME: {item['name']}",
        #             (400, 400),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #             (255, 255, 255), 1)

        cv2.putText(frame, f"LowerLimit: {item['minimum_temp']}",
                    (400, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        cv2.putText(frame, f"AvgTemp: {item['temp_maintain']}",
                    (400, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        cv2.putText(frame, f"UpperLimit: {item['max_temp']}",
                    (400, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

    # # LOWER RIGHT SIDE TEXT
    # cv2.putText(frame, f"NAME: {raw_data["name"]}",
    #             (400, 400),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    #             (255, 255, 255), 1)
    
    # cv2.putText(frame, f"MinTemp: {raw_data["minimum_temp"]}",
    #             (400, 430),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    #             (255, 255, 255), 1)
    
    # cv2.putText(frame, f"AvgTemp: {raw_data["temp_maintain"]}",
    #             (400, 460),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    #             (255, 255, 255), 1)
    
    # cv2.putText(frame, f"MaxTemp: {raw_data["max_temp"]}",
    #             (400, 490),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
    #             (255, 255, 255), 1)

    cv2.imshow("Vision + Modbus", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
modbus_client.close()