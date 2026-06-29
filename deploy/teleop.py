import serial
import time
import ctypes

# ==================== ALL USER VARIABLES ====================
PORT_NAME = 'COM12'
BAUDRATE = 115200
SERIAL_TIMEOUT = 0.02
ARDUINO_WAIT_TIME = 2.0
LOOP_DELAY = 0.02
PRINT_DEVICE_FEEDBACK = True
# ===========================================================

ser = None
user32 = ctypes.windll.user32

VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_F = 0x46
VK_I = 0x49
VK_K = 0x4B
VK_J = 0x4A
VK_L = 0x4C
VK_R = 0x52
VK_Q = 0x51
VK_COMMA = 0xBC
VK_PERIOD = 0xBE

movement_keys = {
    'W': VK_W,
    'A': VK_A,
    'S': VK_S,
    'D': VK_D,
}

edge_keys = {
    'F': VK_F,
    'I': VK_I,
    'K': VK_K,
    'J': VK_J,
    'L': VK_L,
    'R': VK_R,
    ',': VK_COMMA,
    '.': VK_PERIOD,
    'Q': VK_Q,
}


def is_key_down(vk_code):
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


def send_char(ch):
    global ser
    ser.write(ch.encode('utf-8'))


def read_available_lines():
    global ser
    lines = []
    time.sleep(0.01)
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                lines.append(line)
        except Exception:
            pass
    return lines


try:
    ser = serial.Serial(PORT_NAME, BAUDRATE, timeout=SERIAL_TIMEOUT)
    time.sleep(ARDUINO_WAIT_TIME)

    if PRINT_DEVICE_FEEDBACK:
        for line in read_available_lines():
            print(line)

    print("Hold control ready")
    print("Hold W -> climb up")
    print("Hold S -> climb down")
    print("Hold A -> turn left")
    print("Hold D -> turn right")
    print("Release WASD -> stop")
    print("F -> flip up/down")
    print("I/K -> climb speed +/-")
    print("J/L -> turn speed +/-")
    print("R -> flip A/D turn mapping")
    print(",/. -> adjust speed of motors 1/2/3")
    print("Q -> quit")

    prev_edge_state = {k: False for k in edge_keys.keys()}
    press_time = {k: 0.0 for k in movement_keys.keys()}
    last_sent_motion = None

    while True:
        now = time.time()

        # ---------- edge-trigger keys ----------
        for name, vk in edge_keys.items():
            down = is_key_down(vk)
            was_down = prev_edge_state[name]

            if down and not was_down:
                if name == 'Q':
                    raise KeyboardInterrupt
                else:
                    send_char(name)
                    print(f"send: {name}")
                    if PRINT_DEVICE_FEEDBACK:
                        for line in read_available_lines():
                            print(line)

            prev_edge_state[name] = down

        # ---------- hold-trigger movement keys ----------
        current_down = {}
        for name, vk in movement_keys.items():
            down = is_key_down(vk)
            current_down[name] = down
            if down and press_time[name] == 0.0:
                press_time[name] = now
            elif not down:
                press_time[name] = 0.0

        active_motion = None

        held_keys = [k for k, v in current_down.items() if v]

        if len(held_keys) == 0:
            active_motion = '0'
        elif len(held_keys) == 1:
            active_motion = held_keys[0]
        else:
            latest_key = max(held_keys, key=lambda k: press_time[k])
            active_motion = latest_key

        if active_motion != last_sent_motion:
            send_char(active_motion)
            print(f"send: {active_motion}")
            if PRINT_DEVICE_FEEDBACK:
                for line in read_available_lines():
                    print(line)
            last_sent_motion = active_motion

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    pass

except Exception as e:
    print(f"Serial error: {e}")

finally:
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass