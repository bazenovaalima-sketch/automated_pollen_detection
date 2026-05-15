"""
Main entry point for the automated pollen scanner.

Runs two cooperating threads:
  - Scanner thread: drives the X then Y stepper motors and signals the main
    thread to log each new microscope position.
  - Main thread: captures camera frames, runs YOLO/RT-DETR inference, displays
    the annotated feed, and saves a CSV row for each detected object.

Press `q` in the display window to stop cleanly.
"""

import os
import time
import threading

import cv2
import pandas as pd
from pyfirmata import Arduino, util
from ultralytics import RTDETR, YOLO

try:
    from .config import (
        SERIAL_PORT, CAMERA_INDEX,
        X_MOTOR_PINS, Y_MOTOR_PINS,
        MOVES_PER_AXIS, STEPS_PER_MOVE, STEP_DELAY, PAUSE_SECONDS,
        MODEL_TYPE, MODEL_PATH, CONF_THRESHOLD, CAPTURE_DIR, LOG_CSV,
    )
    from .motor_control import attach_motor, release_motor, step_motor
except ImportError:
    from config import (
        SERIAL_PORT, CAMERA_INDEX,
        X_MOTOR_PINS, Y_MOTOR_PINS,
        MOVES_PER_AXIS, STEPS_PER_MOVE, STEP_DELAY, PAUSE_SECONDS,
        MODEL_TYPE, MODEL_PATH, CONF_THRESHOLD, CAPTURE_DIR, LOG_CSV,
    )
    from motor_control import attach_motor, release_motor, step_motor


# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
current_frame      = None              # latest BGR frame from the camera
capture_trigger    = threading.Event() # raised by scanner: log this position
stop_event         = threading.Event() # raised by anyone: shut everything down
current_move_info  = {"label": "X", "id": 0}


# ---------------------------------------------------------------------------
# Camera grabber thread — keeps `current_frame` always up to date
# ---------------------------------------------------------------------------
def camera_worker(cap):
    global current_frame
    while not stop_event.is_set():
        ok, frame = cap.read()
        if ok:
            current_frame = frame
        else:
            time.sleep(0.01)


# ---------------------------------------------------------------------------
# Scanner thread — moves the motors, then waits for the main loop to log
# ---------------------------------------------------------------------------
def scanner_worker(x_motor, y_motor):
    try:
        # --- PHASE 1: X axis ---
        print("\n--- PHASE 1: MOVING X MOTOR ---")
        for i in range(MOVES_PER_AXIS):
            if stop_event.is_set():
                break
            print(f"\n[X Move {i+1}/{MOVES_PER_AXIS}] Moving...")
            step_motor(x_motor, STEPS_PER_MOVE, STEP_DELAY)

            print(f"Waiting {PAUSE_SECONDS}s for focus/settle...")
            time.sleep(PAUSE_SECONDS)

            current_move_info["label"] = "X"
            current_move_info["id"]    = i + 1
            capture_trigger.set()
            while capture_trigger.is_set() and not stop_event.is_set():
                time.sleep(0.05)

        # --- PHASE 2: Y axis ---
        print("\n--- PHASE 2: MOVING Y MOTOR ---")
        for i in range(MOVES_PER_AXIS):
            if stop_event.is_set():
                break
            print(f"\n[Y Move {i+1}/{MOVES_PER_AXIS}] Moving...")
            step_motor(y_motor, STEPS_PER_MOVE, STEP_DELAY)

            print(f"Waiting {PAUSE_SECONDS}s for focus/settle...")
            time.sleep(PAUSE_SECONDS)

            current_move_info["label"] = "Y"
            current_move_info["id"]    = i + 1
            capture_trigger.set()
            while capture_trigger.is_set() and not stop_event.is_set():
                time.sleep(0.05)

        print("\n✅ All moves complete.")

    except Exception as exc:
        print(f"Scanner error: {exc}")
    finally:
        stop_event.set()


def load_detector(model_type, model_path):
    """Load the detector selected in config.py."""
    if model_type.lower() == "rtdetr":
        return RTDETR(model_path)
    return YOLO(model_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    # Arduino / motors
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    board = Arduino(SERIAL_PORT)
    it = util.Iterator(board)
    it.start()
    x_motor = attach_motor(board, X_MOTOR_PINS)
    y_motor = attach_motor(board, Y_MOTOR_PINS)

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # Model
    print(f"Loading {MODEL_TYPE} model from {MODEL_PATH}...")
    model = load_detector(MODEL_TYPE, MODEL_PATH)

    # Threads
    cam_thread  = threading.Thread(target=camera_worker, args=(cap,), daemon=True)
    scan_thread = threading.Thread(target=scanner_worker, args=(x_motor, y_motor), daemon=True)
    cam_thread.start()
    scan_thread.start()

    results_log = []
    print("🚀 Live feed with AI starting. Motors will follow.")

    try:
        while not stop_event.is_set():
            if current_frame is not None:
                # ---- live inference ----
                results = model.predict(source=current_frame,
                                        conf=CONF_THRESHOLD,
                                        verbose=False)[0]
                annotated = results.plot()
                cv2.imshow("ARM-2 Scanner Live View", annotated)

                # ---- scanner asked us to log this position ----
                if capture_trigger.is_set():
                    n = len(results.boxes)
                    label = current_move_info["label"]
                    move_id = current_move_info["id"]

                    if n > 0:
                        print(f"✅ Found {n} objects at {label}{move_id}")
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        img_path = f"{CAPTURE_DIR}/detect_{ts}_{label}{move_id}.jpg"
                        cv2.imwrite(img_path, annotated)

                        for box in results.boxes:
                            results_log.append({
                                "Timestamp" : time.strftime("%H:%M:%S"),
                                "Move_Type" : label,
                                "Move_ID"   : move_id,
                                "Label"     : model.names[int(box.cls[0])],
                                "Confidence": f"{float(box.conf[0]):.2f}",
                                "Image"     : img_path,
                            })
                        pd.DataFrame(results_log).to_csv(LOG_CSV, index=False)
                    else:
                        print(f"⚪️ No detections at {label}{move_id}")

                    capture_trigger.clear()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up...")
        stop_event.set()
        release_motor(x_motor)
        release_motor(y_motor)
        board.exit()
        cap.release()
        cv2.destroyAllWindows()
        print("Closed.")


if __name__ == "__main__":
    main()
