from pymodbus.client import ModbusSerialClient

# Create Modbus RTU client
client = ModbusSerialClient(
    port='COM6',        # बदलो (e.g. COM3 or /dev/ttyUSB0)
    baudrate=9600,
    stopbits=1,
    bytesize=8,
    parity='N',
    timeout=1
)

# Connect
if not client.connect():
    print("Unable to connect")
    exit()

# Modbus slave ID
slave_id = 1

# Address conversion
start_address = 0x41000          # 1000h = 4096
num_registers = (0x41022 - 0x41000) + 1   # inclusive

# Read Holding Registers (Function Code 03)
response = client.read_holding_registers(
    address=start_address,
    count=num_registers
)

# Check response
if response.isError():
    print("Error:", response)
else:
    print("Register Values:")
    for i, val in enumerate(response.registers):
        print(f"Address {hex(start_address + i)} : {val}")

# Close connection
client.close()
