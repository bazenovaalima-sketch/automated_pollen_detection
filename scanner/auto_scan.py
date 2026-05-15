import os
import time
import threading

import cv2
import pandas as pd
from pyfirmata import Arduino, util
from ultralytics import RTDETR, YOLO

try:
    from .config import (
        SERIAL_PORT,
        CAMERA_INDEX,
        X_MOTOR_PINS,
        Y_MOTOR_PINS,
        MOVES_PER_AXIS,
        STEPS_PER_MOVE,
        STEP_DELAY,
        PAUSE_SECONDS,
        MODEL_TYPE,
        MODEL_PATH,
        CONF_THRESHOLD,
        CAPTURE_DIR,
        LOG_CSV,
    )
    from .motor_control import attach_motor, release_motor, step_motor
except ImportError:
    from config import (
        SERIAL_PORT,
        CAMERA_INDEX,
        X_MOTOR_PINS,
        Y_MOTOR_PINS,
        MOVES_PER_AXIS,
        STEPS_PER_MOVE,
        STEP_DELAY,
        PAUSE_SECONDS,
        MODEL_TYPE,
        MODEL_PATH,
        CONF_THRESHOLD,
        CAPTURE_DIR,
        LOG_CSV,
    )
    from motor_control import attach_motor, release_motor, step_motor


current_frame = None
frame_lock = threading.Lock()

capture_trigger = threading.Event()
stop_event = threading.Event()

current_move_info = {
    "axis": "X",
    "move_id": 0,
}


def camera_worker(cap):
    global current_frame

    while not stop_event.is_set():
        success, frame = cap.read()

        if not success:
            time.sleep(0.01)
            continue

        with frame_lock:
            current_frame = frame.copy()


def scanner_worker(x_motor, y_motor):
    try:
        scan_axis("X", x_motor)
        scan_axis("Y", y_motor)
    except Exception as error:
        print(f"Scanner error: {error}")
    finally:
        stop_event.set()


def scan_axis(axis_name, motor):
    print(f"Starting {axis_name}-axis scan")

    for move_index in range(1, MOVES_PER_AXIS + 1):
        if stop_event.is_set():
            break

        print(f"{axis_name}-axis move {move_index}/{MOVES_PER_AXIS}")

        step_motor(
            motor,
            STEPS_PER_MOVE,
            STEP_DELAY,
        )

        time.sleep(PAUSE_SECONDS)

        current_move_info["axis"] = axis_name
        current_move_info["move_id"] = move_index

        capture_trigger.set()

        while capture_trigger.is_set() and not stop_event.is_set():
            time.sleep(0.05)


def load_detector(model_type, model_path):
    if model_type.lower() == "rtdetr":
        return RTDETR(model_path)

    return YOLO(model_path)


def initialize_arduino():
    print(f"Connecting to Arduino on {SERIAL_PORT}")

    board = Arduino(SERIAL_PORT)

    iterator = util.Iterator(board)
    iterator.start()

    x_motor = attach_motor(board, X_MOTOR_PINS)
    y_motor = attach_motor(board, Y_MOTOR_PINS)

    return board, x_motor, y_motor


def initialize_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    return cap


def save_detections(results, model, annotated_frame, results_log):
    axis = current_move_info["axis"]
    move_id = current_move_info["move_id"]
    detection_count = len(results.boxes)

    if detection_count == 0:
        print(f"No detections at {axis}{move_id}")
        return

    print(f"Found {detection_count} objects at {axis}{move_id}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(
        CAPTURE_DIR,
        f"detect_{timestamp}_{axis}{move_id}.jpg",
    )

    cv2.imwrite(image_path, annotated_frame)

    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        results_log.append(
            {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Move_Type": axis,
                "Move_ID": move_id,
                "Label": model.names[class_id],
                "Confidence": f"{confidence:.2f}",
                "Image": image_path,
            }
        )

    pd.DataFrame(results_log).to_csv(LOG_CSV, index=False)


def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    board = None
    cap = None
    x_motor = None
    y_motor = None

    try:
        board, x_motor, y_motor = initialize_arduino()
        cap = initialize_camera()

        print(f"Loading {MODEL_TYPE} model from {MODEL_PATH}")
        model = load_detector(MODEL_TYPE, MODEL_PATH)

        camera_thread = threading.Thread(
            target=camera_worker,
            args=(cap,),
            daemon=True,
        )

        scanner_thread = threading.Thread(
            target=scanner_worker,
            args=(x_motor, y_motor),
            daemon=True,
        )

        camera_thread.start()
        scanner_thread.start()

        results_log = []

        print("Scanner started")

        while not stop_event.is_set():
            with frame_lock:
                frame = None if current_frame is None else current_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                verbose=False,
            )[0]

            annotated_frame = results.plot()
            cv2.imshow("Pollen Scanner Live View", annotated_frame)

            if capture_trigger.is_set():
                save_detections(
                    results,
                    model,
                    annotated_frame,
                    results_log,
                )

                capture_trigger.clear()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        print("Cleaning up")

        stop_event.set()

        if x_motor is not None:
            release_motor(x_motor)

        if y_motor is not None:
            release_motor(y_motor)

        if board is not None:
            board.exit()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        print("Closed")


if __name__ == "__main__":
    main()
