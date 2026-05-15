"""Central configuration for the automated pollen scanner."""

# ---- Serial / camera ------------------------------------------------------
SERIAL_PORT   = "/dev/cu.usbmodem14101"   # macOS example; Windows: "COM3"
CAMERA_INDEX  = 0                          # OpenCV camera index

# ---- Motor pin map (Arduino MEGA, via Firmata) ----------------------------
X_MOTOR_PINS  = [4, 5, 6, 7]               # IN1..IN4 of the X-axis ULN2003
Y_MOTOR_PINS  = [8, 9, 10, 11]             # IN1..IN4 of the Y-axis ULN2003

# ---- Scan pattern ---------------------------------------------------------
MOVES_PER_AXIS = 10                        # stops per axis
STEPS_PER_MOVE = 200                       # micro-steps between stops
STEP_DELAY     = 0.002                     # seconds between micro-steps
PAUSE_SECONDS  = 2.0                       # settle / focus pause after move

# ---- Inference ------------------------------------------------------------
# Default deployment model: YOLOv8l. Change MODEL_TYPE to "rtdetr" if you want
# to test the RT-DETR checkpoint instead.
MODEL_TYPE      = "yolo"
MODEL_PATH      = "models/weights/best_yolo8.pt"
CONF_THRESHOLD  = 0.30
CAPTURE_DIR     = "captures"
LOG_CSV         = "auto_scan_log.csv"
