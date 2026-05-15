# Arduino

This project drives the Arduino MEGA 2560 directly from Python via the [Firmata](https://github.com/firmata/protocol) protocol, so **no custom Arduino sketch is required**.

## Flashing StandardFirmata

1. Plug the Arduino MEGA into your computer.
2. Open the Arduino IDE.
3. Go to **File → Examples → Firmata → StandardFirmata**.
4. Select **Tools → Board → Arduino MEGA or MEGA 2560**.
5. Select the correct **Port** under **Tools → Port**.
6. Click **Upload** (→).

Once flashed, the Python module `scanner/auto_scan.py` will connect over the same serial port. Set the port in `scanner/config.py` as `SERIAL_PORT`.
