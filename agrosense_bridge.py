"""
AgroSense — Serial Bridge
Reads ESP32 sensor data over USB serial and serves it
as a local HTTP API that the HTML dashboard can fetch from.

INSTALL (run once in Command Prompt):
    pip install pyserial flask flask-cors

RUN:
    python agrosense_bridge.py

Then open barn_fish_ui.html in your browser.
"""

import serial
import json
import threading
from flask import Flask, jsonify
from flask_cors import CORS

# ── SETTINGS — change COM port to match your ESP32 ──────────
SERIAL_PORT = "COM5"   # Change this! Check Arduino IDE → Tools → Port
BAUD_RATE   = 115200
# ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # allows HTML file to fetch from localhost

# Latest sensor reading (shared between serial thread and Flask)
latest = {
    "temp":  25.0,
    "hum":   60.0,
    "gas":   100,
    "sound": 0
}

def read_serial():
    """Continuously reads JSON lines from ESP32 over serial."""
    global latest
    print(f"Connecting to ESP32 on {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print("Connected! Reading sensor data...")
        while True:
            try:
                line = ser.readline().decode("utf-8").strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    if "error" not in data:
                        latest = data
                        print(f"Temp:{data['temp']}°C  Hum:{data['hum']}%  Gas:{data['gas']}  Sound:{data['sound']}")
            except json.JSONDecodeError:
                pass  # skip malformed lines
            except Exception as e:
                print(f"Read error: {e}")
    except serial.SerialException as e:
        print(f"\nERROR: Could not open {SERIAL_PORT}")
        print("Fix: Change SERIAL_PORT in this script to match Arduino IDE → Tools → Port")
        print(f"Details: {e}")

@app.route("/sensors")
def sensors():
    """HTML dashboard fetches this endpoint every 2.5 seconds."""
    return jsonify(latest)

@app.route("/")
def index():
    return "AgroSense Bridge running! Dashboard can fetch from http://localhost:5000/sensors"

if __name__ == "__main__":
    # Start serial reading in background thread
    t = threading.Thread(target=read_serial, daemon=True)
    t.start()

    print("\nAgroSense Bridge started!")
    print("Dashboard endpoint: http://localhost:5000/sensors")
    print("Keep this window open while using the dashboard.\n")

    app.run(host="0.0.0.0", port=5000, debug=False)