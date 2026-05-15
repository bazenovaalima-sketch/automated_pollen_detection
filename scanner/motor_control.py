"""
motor_control.py
================
Thin helpers around the 28BYJ-48 + ULN2003 stepper driven through pyFirmata.

We use a standard half-step sequence (8 sub-steps per cycle) for smoother
motion than the simpler 4-step full-step sequence.
"""

import time

# Half-step sequence for a 4-coil stepper (IN1..IN4).
HALF_STEP_SEQUENCE = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]


def attach_motor(board, pins):
    """Return a list of pyFirmata digital-output pin objects."""
    return [board.get_pin(f"d:{p}:o") for p in pins]


def release_motor(motor_pins):
    """De-energise all coils — important to avoid the motor heating up."""
    for pin in motor_pins:
        pin.write(0)


def step_motor(motor_pins, total_steps, step_delay=0.002, direction=1):
    """
    Drive a stepper for `total_steps` half-steps.

    Parameters
    ----------
    motor_pins  : list of 4 pyFirmata digital-output pins
    total_steps : int   — positive number of half-steps to execute
    step_delay  : float — seconds between successive half-steps
    direction   : +1 forward, -1 reverse
    """
    seq = HALF_STEP_SEQUENCE if direction > 0 else list(reversed(HALF_STEP_SEQUENCE))
    for i in range(total_steps):
        pattern = seq[i % len(seq)]
        for pin, value in zip(motor_pins, pattern):
            pin.write(value)
        time.sleep(step_delay)
    release_motor(motor_pins)
