import cv2
import json
import time
from pymodbus.client import ModbusSerialClient
from ultralytics import YOLO  # <--- Added Ultralytics

# ==========================================
# 1. CONFIGURATION & MAPPING
# ==========================================
JSON_MAP_FILE = "item_temps.json"

# RS485 / Modbus Settings
RS485_PORT = 'COM3'  # Update to your USB RS485 port
BAUD_RATE = 9600
PID_SLAVE_ID = 1

READ_LOCATIONS = [0x1000, 0x1001, 0x1002]  
WRITE_LOCATIONS = [0x1003, 0x1004, 0x1005] 

# Vision Model Settings - Now using the PyTorch file directly!
MODEL_PT = "best.pt"

# ==========================================
# 2. INITIALIZATION
# ==========================================
# Load the item-to-temperature map
try:
    with open(JSON_MAP_FILE, 'r') as f:
        item_temp_map = json.load(f)
except FileNotFoundError:
    print(f"Warning: {JSON_MAP_FILE} not found. Creating a dummy file.")
    item_temp_map = {"TargetObject": [150, 160, 170]} # Update "TargetObject" to your actual class name
    with open(JSON_MAP_FILE, 'w') as f:
        json.dump(item_temp_map, f, indent=4)

# Initialize Modbus RS485 Client
modbus_client = ModbusSerialClient(
    port=RS485_PORT, 
    baudrate=BAUD_RATE, 
    timeout=0.05, 
    stopbits=1, 
    bytesize=8, 
    parity='N'
)

# Initialize Ultralytics YOLO model
print(f"Loading {MODEL_PT} model...")
model = YOLO(MODEL_PT)

# Initialize Camera
cap = cv2.VideoCapture(0)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def read_pid_data():
    data = []
    comm_status = False
    if modbus_client.connect():
        try:
            for addr in READ_LOCATIONS:
                result = modbus_client.read_holding_registers(address=addr, count=1, slave=PID_SLAVE_ID)
                if not result.isError():
                    data.append(result.registers[0])
                    comm_status = True
                else:
                    data.append("ERR")
                    comm_status = False
        except Exception as e:
            comm_status = False
            data = ["ERR", "ERR", "ERR"]
    return data, comm_status

def write_pid_data(temp_values):
    if modbus_client.connect():
        for i, addr in enumerate(WRITE_LOCATIONS):
            if i < len(temp_values):
                try:
                    modbus_client.write_register(address=addr, value=temp_values[i], slave=PID_SLAVE_ID)
                except Exception as e:
                    print(f"Modbus Write Error at {hex(addr)}: {e}")

# ==========================================
# 4. MAIN LIVE LOOP
# ==========================================
last_detected_item = None
current_temperatures = ["N/A", "N/A", "N/A"]
comm_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- A. Computer Vision Detection (Ultralytics) ---
    # verbose=False stops it from printing detection logs every single frame
    results = model(frame, verbose=False)
    
    detected_item_this_frame = None

    # Parse Ultralytics results
    for result in results:
        boxes = result.boxes

    if boxes is None:
        continue  # No detections in this frame

    for box in boxes:
        conf = box.conf[0].item()

        if conf > 0.5:
            class_id = int(box.cls[0].item())
            detected_item_this_frame = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{detected_item_this_frame} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            break
    # for result in results:
    #     boxes = result.boxes
    #     for box in boxes:
    #         conf = box.conf[0].item()  # Confidence score
            
    #         if conf > 0.5:
    #             class_id = int(box.cls[0].item())
    #             detected_item_this_frame = model.names[class_id] # Gets the actual class name string!
                
    #             # Bounding box coordinates
    #             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    #             cv2.putText(frame, f"{detected_item_this_frame} {conf:.2f}", (x1, y1 - 10), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
    #             # We only need the first confident detection to trigger the PID
    #             break 

    # --- B. RS485 Logic Trigger ---
    if detected_item_this_frame and detected_item_this_frame != last_detected_item:
        if detected_item_this_frame in item_temp_map:
            new_set_temps = item_temp_map[detected_item_this_frame]
            print(f"Detected {detected_item_this_frame}. Pushing new temps to PID: {new_set_temps}")
            write_pid_data(new_set_temps)
        else:
            print(f"Detected {detected_item_this_frame}, but it's not in the JSON map file.")
        
        last_detected_item = detected_item_this_frame

    current_temperatures, comm_active = read_pid_data()

    # --- C. Live Feed GUI Overlay ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    status_color = (0, 255, 0) if comm_active else (0, 0, 255)
    cv2.circle(frame, (35, 35), 8, status_color, -1)
    cv2.putText(frame, "RS485 Link", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Last Detected: {last_detected_item or 'None'}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Loc 1 Actual Temp: {current_temperatures[0]}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Loc 2 Actual Temp: {current_temperatures[1]}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Loc 3 Actual Temp: {current_temperatures[2]}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Industrial Vision & PID Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
modbus_client.close()