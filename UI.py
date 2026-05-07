import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import time
import subprocess
import json
import shutil
import threading
from datetime import datetime

import RPi.GPIO as GPIO
import cv2
import pygame
from evdev import InputDevice, ecodes, list_devices
from picamera2 import Picamera2
import numpy as np


# ============================
# Settings
# ============================
SCREEN_W = 800
SCREEN_H = 480
LOGO_PATH = "/home/atlas/atlas_project/atlaslogo.png"

FONT_BOLD = "/home/atlas/atlas_project/fonts/AtlasFontBold.otf"
FONT_REG = "/home/atlas/atlas_project/fonts/AtlasFontRegular.otf"
FONT_LIGHT = "/home/atlas/atlas_project/fonts/OrbitronFont.ttf"

SHOW_SPLASH = True
SPLASH_SECONDS = 4
DEBUG_TOUCH = True

LED_PWM_PIN = 25
LED_PWM_FREQ = 8000
FOCUS_LIGHT_DUTY = 20
FOCUS_LIGHT_DURATION = 3.0

LEPTON_VIDEO_DEVICE = "/dev/video0"

SCREENSHOT_DIR = "/home/atlas/screenshots"
TRIGGER_FILE = "/tmp/take_screenshot"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Set this manually to the actual path of your trained YOLO weights on the Pi.
YOLO_MODEL_PATH = "/home/atlas/atlas_project/best.pt"
YOLO_CONF_THRESHOLD = 0.25
USE_COLOUR_PREPROCESSING_FOR_YOLO = False

# Colour image capture lighting.
PLATE_CAPTURE_LIGHT_DUTY = 35
PLATE_CAPTURE_LIGHT_SECONDS = 3.0

# Colour image review timing
review_start_time = None
REVIEW_AUTO_SECONDS = 8.0

# Thermal/radiometric settings.
# PureThermal exposed Y16 in v4l2-ctl, so this code requests Y16 via OpenCV/V4L2.
# Radiometric Lepton frames are usually Kelvin x100, e.g. 30000 = 300.00 K = 26.85 C.
LEPTON_RAW_SCALE = 100.0
LEPTON_KELVIN_OFFSET = 273.15
THERMAL_CAL_A = 1.0
THERMAL_CAL_B = 0.0
THERMAL_PREVIEW_RANGE_C = 5.0
THERMAL_CENTRE_ROI_FRACTION = 0.60
THERMAL_COOL_LEAF_PERCENTILE = 35
USE_THERMAL_LEAF_MASK = True
THERMAL_DISPLAY_MIN_C = 16.0
THERMAL_DISPLAY_MAX_C = 26.0
USE_FIXED_THERMAL_DISPLAY_RANGE = True

X_LIMIT_PIN = 5
Z_LIMIT_PIN = 6

# X 3-wire switch. Change to GPIO.HIGH if it reads HIGH when pressed.
X_LIMIT_PRESSED_STATE = GPIO.LOW

# Z NC switch:
# unpressed = grounded = LOW
# pressed = open + internal pull-up = HIGH
# If your UI is backwards, change this to GPIO.HIGH.
Z_LIMIT_PRESSED_STATE = GPIO.LOW

USB_MOUNT_PATH = "/mnt/atlas_usb"

LOAD_GUIDE_VIDEO_PATH = "/home/atlas/atlas_project/Atlas_load_clip.mp4"
guide_video_cap = None

CAMERA_LOGO_PATH = "/home/atlas/atlas_project/cameralogo.png"

# ============================
# Colours
# ============================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (120, 220, 120)
RED = (230, 120, 120)
BLUE = (170, 220, 255)
GREY = (220, 220, 220)
DARK_GREY = (60, 60, 60)
YELLOW = (240, 220, 120)


# ============================================================
# COMMENTED OUT: Stepper Motor HAT (B) / DRV8825 configuration
# Keep this here for when the Waveshare HAT is replaced.
# ============================================================

# # Motor 1 = X axis
# X_DIR_PIN = 13
# X_STEP_PIN = 19
# X_ENABLE_PIN = 12
#
# # Motor 2 = Z axis
# Z_DIR_PIN = 24
# Z_STEP_PIN = 18
# Z_ENABLE_PIN = 4
#
# X_DIR_RIGHT = GPIO.HIGH
# X_DIR_LEFT = GPIO.LOW
# Z_DIR_UP = GPIO.HIGH
# Z_DIR_DOWN = GPIO.LOW
#
# X_STEPS_PER_MM = 10
# Z_STEPS_PER_MM = 10
# X_MAX_MM = 20.0
# Z_MAX_MM = 20.0
#
# STEP_PULSE_SECONDS = 0.0008
# STEP_GAP_SECONDS = 0.0008
# AUTO_HOME_ON_STARTUP = False


# ============================================================
# RESERVED FOR SETTINGS SCREEN ONLY: L298N stepper motor config
# This is not used by the main scan workflow. It only powers the manual
# motor-control page under Settings > Motor Control.
# ============================================================

# X-axis L298N input pins: IN1, IN2, IN3, IN4
X_MOTOR_PINS = [13, 19, 16, 20]

# Z-axis L298N input pins: IN1, IN2, IN3, IN4
Z_MOTOR_PINS = [24, 18, 4, 17]

# Temporary calibration values. Start low for safe first tests.
X_STEPS_PER_MM = 5
Z_STEPS_PER_MM = 5

# Temporary software travel limits. Replace later with measured values.
X_MAX_MM = 100.0
Z_MAX_MM = 100.0

# Delay between half-steps. Larger = slower but safer/more torque.
STEPPER_DELAY = 0.006

# Maximum homing half-steps before giving up.
X_HOME_MAX_STEPS = 50
Z_HOME_MAX_STEPS = 50


# ============================
# Init pygame
# ============================
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.FULLSCREEN)
pygame.display.set_caption("ATLAS")
logical_surface = pygame.Surface((SCREEN_W, SCREEN_H))


# ============================
# App state
# ============================
state = "home"
running = True

# Motor state - Settings > Motor Control only
x_pos_mm = 0.0
z_pos_mm = 0.0
motors_homed = False
motor_busy = False
motor_status_text = "NOT HOMED"
motor_status_colour = RED

picam2 = None
focus_camera_active = False

focus_light_on = False
focus_light_off_time = None

thermal_cap = None
thermal_active = False
thermal_last_dtype = None
thermal_last_min = None
thermal_last_max = None

x_limit_pressed = False
z_limit_pressed = False
prev_x_limit_pressed = False
prev_z_limit_pressed = False

status_text = "READY"
status_colour = WHITE

current_scan_folder = None
selected_seedling = None
run_start_time = None

# Manual scan workflow state
plate_image_rgb = None
plate_annotated_rgb = None
seedling_detections = []
current_seedling_index = 0
latest_display_boxes = []
yolo_model = None
yolo_model_load_attempted = False

# Placeholder results kept as fallback if no real scan has been completed.
fake_seedlings = [
    {"id": 1, "temp": 24.6, "rect": pygame.Rect(170, 120, 110, 80)},
    {"id": 2, "temp": 25.1, "rect": pygame.Rect(345, 105, 110, 85)},
    {"id": 3, "temp": 24.9, "rect": pygame.Rect(520, 160, 100, 90)},
]


# ============================
# GPIO setup
# ============================
led_pwm = None

try:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # LED PWM output
    GPIO.setup(LED_PWM_PIN, GPIO.OUT)
    led_pwm = GPIO.PWM(LED_PWM_PIN, LED_PWM_FREQ)
    led_pwm.start(0)

    # Limit switches
    # X: 3-wire powered switch. No pull-up/down because the module drives signal.
    GPIO.setup(X_LIMIT_PIN, GPIO.IN)

    # Z: 2-wire NC switch to GND. Pull-up makes it HIGH when switch opens.
    GPIO.setup(Z_LIMIT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # ============================================================
    # COMMENTED OUT: Stepper Motor HAT GPIO setup
    # ============================================================
    # GPIO.setup(X_DIR_PIN, GPIO.OUT)
    # GPIO.setup(X_STEP_PIN, GPIO.OUT)
    # GPIO.setup(X_ENABLE_PIN, GPIO.OUT)
    # GPIO.setup(Z_DIR_PIN, GPIO.OUT)
    # GPIO.setup(Z_STEP_PIN, GPIO.OUT)
    # GPIO.setup(Z_ENABLE_PIN, GPIO.OUT)
    # GPIO.output(X_STEP_PIN, GPIO.LOW)
    # GPIO.output(Z_STEP_PIN, GPIO.LOW)
    # GPIO.output(X_ENABLE_PIN, GPIO.LOW)  # DRV8825 enable is active LOW
    # GPIO.output(Z_ENABLE_PIN, GPIO.LOW)

    # ============================================================
    # RESERVED: L298N motor output pins for Settings > Motor Control only
    # ============================================================
    for pin in X_MOTOR_PINS + Z_MOTOR_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    print("GPIO initialised OK")

except Exception as e:
    print("GPIO init failed:", e)
    led_pwm = None


# ============================
# Fonts
# ============================
font_title = pygame.font.Font(FONT_BOLD, 60)
font_button = pygame.font.Font(FONT_LIGHT, 36)
font_status = pygame.font.Font(FONT_LIGHT, 34)
font_body = pygame.font.Font(FONT_REG, 30)
font_small = pygame.font.Font(FONT_LIGHT, 24)


# ============================
# Touch state
# ============================
touch_x = None
touch_y = None
touch_pressed = False


# ============================
# Rects
# ============================
start_rect = pygame.Rect(220, 170, 360, 95)

settings_rect = pygame.Rect(20, 400, 170, 50)
settings_close_rect = pygame.Rect(725, 20, 50, 50)
focus_button_rect = pygame.Rect(200, 130, 400, 80)
thermal_button_rect = pygame.Rect(200, 225, 400, 80)
focus_close_rect = pygame.Rect(725, 20, 50, 50)
focus_light_rect = pygame.Rect(20, 180, 60, 60)

motor_button_rect = pygame.Rect(200, 320, 400, 80)
motor_close_rect = pygame.Rect(725, 20, 50, 50)
home_motor_rect = pygame.Rect(340, 190, 120, 80)

x_left_10_rect = pygame.Rect(70, 200, 85, 60)
x_left_1_rect = pygame.Rect(165, 200, 85, 60)
x_right_1_rect = pygame.Rect(550, 200, 85, 60)
x_right_10_rect = pygame.Rect(645, 200, 85, 60)

z_up_10_rect = pygame.Rect(360, 75, 80, 55)
z_up_1_rect = pygame.Rect(360, 135, 80, 45)
z_down_1_rect = pygame.Rect(360, 285, 80, 45)
z_down_10_rect = pygame.Rect(360, 340, 80, 55)

yes_rect = pygame.Rect(120, 340, 220, 85)
no_rect = pygame.Rect(460, 340, 220, 85)

help_close_rect = pygame.Rect(725, 20, 50, 50)
stop_small_rect = pygame.Rect(650, 20, 120, 55)

result_next_rect = pygame.Rect(645, 400, 130, 50)
detail_close_rect = pygame.Rect(725, 20, 50, 50)

discard_rect = pygame.Rect(90, 330, 280, 85)
save_rect = pygame.Rect(430, 330, 280, 85)
save_done_rect = pygame.Rect(190, 330, 420, 85)

# Manual workflow buttons
colour_live_view_rect = pygame.Rect(20, 65, 600, 365)
colour_capture_rect = pygame.Rect(600, 100, 170, 270)
colour_retake_rect = pygame.Rect(40, 400, 220, 60)
edit_boxes_rect = pygame.Rect(310, 400, 180, 60)
edit_done_rect = pygame.Rect(590, 400, 180, 60)
edit_clear_rect = pygame.Rect(30, 400, 180, 60)
colour_confirm_rect = pygame.Rect(540, 400, 220, 60)
manual_ready_rect = pygame.Rect(220, 330, 360, 80)
guide_back_rect = pygame.Rect(40, 400, 180, 60)
guide_thermal_rect = pygame.Rect(580, 400, 180, 60)
thermal_back_rect = pygame.Rect(40, 400, 200, 60)
thermal_capture_rect = pygame.Rect(560, 400, 200, 60)

# Hidden maintenance exit hotspot (home screen only)
hidden_exit_rect = pygame.Rect(0, 0, 80, 80)
hidden_exit_hold_start = None
HIDDEN_EXIT_HOLD_SECONDS = 4.0


# ============================
# Basic drawing helpers
# ============================
def draw_text_center(surface, text, font, colour, center):
    text_surface = font.render(text, True, colour)
    text_rect = text_surface.get_rect(center=center)
    surface.blit(text_surface, text_rect)


def draw_button(surface, rect, colour, text, text_font, text_colour):
    pygame.draw.rect(surface, colour, rect, border_radius=14)
    label = text_font.render(text, True, text_colour)
    label_rect = label.get_rect(center=rect.center)
    surface.blit(label, label_rect)


def draw_image_button(surface, rect, colour, image_path):
    pygame.draw.rect(surface, colour, rect, border_radius=18)

    if os.path.exists(image_path):
        try:
            icon = pygame.image.load(image_path).convert_alpha()
            iw, ih = icon.get_size()
            scale = min((rect.width * 0.65) / iw, (rect.height * 0.65) / ih)
            new_size = (int(iw * scale), int(ih * scale))
            icon = pygame.transform.smoothscale(icon, new_size)

            x = rect.x + (rect.width - new_size[0]) // 2
            y = rect.y + (rect.height - new_size[1]) // 2
            surface.blit(icon, (x, y))
        except Exception as e:
            print("Button image failed:", e)
            draw_text_center(surface, "CAP", font_small, BLACK, rect.center)
    else:
        draw_text_center(surface, "CAP", font_small, BLACK, rect.center)


def draw_x_button(surface, rect):
    pygame.draw.rect(surface, (60, 60, 60), rect, border_radius=10)
    pygame.draw.rect(surface, (255, 255, 255), rect, width=2, border_radius=10)

    font = pygame.font.Font(FONT_LIGHT, 28)
    label = font.render("X", True, (255, 255, 255))
    label_rect = label.get_rect(center=rect.center)
    surface.blit(label, label_rect)


def draw_rgb_image_fit(surface, image_rgb, max_rect, border=True):
    """Draw an RGB numpy image fitted inside max_rect. Returns (x, y, w, h, scale)."""
    if image_rgb is None:
        return None

    img_h, img_w = image_rgb.shape[:2]
    scale = min(max_rect.width / img_w, max_rect.height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    x = max_rect.x + (max_rect.width - new_w) // 2
    y = max_rect.y + (max_rect.height - new_h) // 2

    resized = cv2.resize(image_rgb, (new_w, new_h))
    img_surface = pygame.surfarray.make_surface(resized.swapaxes(0, 1))
    surface.blit(img_surface, (x, y))
    if border:
        pygame.draw.rect(surface, WHITE, (x, y, new_w, new_h), 2)
    return x, y, new_w, new_h, scale


# ============================
# Touch helpers
# ============================
def find_touch_device():
    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name.lower()
        if "touch" in name or "ft5x06" in name:
            return path
    raise RuntimeError("No touchscreen device found")


def map_range(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def touch_thread():
    global touch_x, touch_y, touch_pressed, running

    try:
        dev = InputDevice(find_touch_device())
        caps = dev.capabilities(absinfo=True)

        abs_x_info = caps[ecodes.EV_ABS][0][1]
        abs_y_info = caps[ecodes.EV_ABS][1][1]

        x_min, x_max = abs_x_info.min, abs_x_info.max
        y_min, y_max = abs_y_info.min, abs_y_info.max

        raw_x = 0
        raw_y = 0

        for event in dev.read_loop():
            if not running:
                break

            if event.type == ecodes.EV_ABS:
                if event.code == ecodes.ABS_X:
                    raw_x = event.value
                elif event.code == ecodes.ABS_Y:
                    raw_y = event.value

                mapped_x = map_range(raw_x, x_min, x_max, 0, SCREEN_W)
                mapped_y = map_range(raw_y, y_min, y_max, 0, SCREEN_H)

                touch_x = SCREEN_W - mapped_x
                touch_y = SCREEN_H - mapped_y

            elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TOUCH:
                touch_pressed = (event.value == 1)

    except Exception as e:
        print("Touch thread error:", e)


# ============================================================
# USB helpers
# ============================================================
def find_usb_partition():
    try:
        result = subprocess.run(
            ["lsblk", "-J", "-o", "NAME,TRAN,TYPE"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    for device in data.get("blockdevices", []):
        is_usb = device.get("tran") == "usb"
        is_disk = device.get("type") == "disk"

        if is_usb and is_disk:
            for child in device.get("children", []):
                if child.get("type") == "part":
                    return f"/dev/{child['name']}"

    return None


def usb_inserted():
    return find_usb_partition() is not None


def usb_mounted():
    return os.path.ismount(USB_MOUNT_PATH)


def mount_usb():
    usb_device = find_usb_partition()
    if usb_device is None:
        return False

    os.makedirs(USB_MOUNT_PATH, exist_ok=True)

    if usb_mounted():
        return True

    uid = os.getuid()
    gid = os.getgid()

    # Requires sudoers rule:
    # atlas ALL=(ALL) NOPASSWD: /usr/bin/mount, /usr/bin/umount
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "mount",
            "-o",
            f"uid={uid},gid={gid},rw",
            usb_device,
            USB_MOUNT_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        print("USB mount failed:", result.stderr)

    return result.returncode == 0


def unmount_usb():
    if usb_mounted():
        subprocess.run(
            ["sudo", "-n", "umount", USB_MOUNT_PATH],
            capture_output=True,
            text=True,
            timeout=10,
        )


def get_usb_path():
    if usb_mounted():
        return USB_MOUNT_PATH
    return None


def create_scan_folder():
    try:
        usb_path = get_usb_path()
        if usb_path is None:
            return None

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(usb_path, f"atlas_scan_{timestamp}")
        os.makedirs(folder, exist_ok=False)
        return folder
    except Exception as e:
        print(f"create_scan_folder failed: {e}")
        return None


def delete_scan_folder(folder_path):
    try:
        if folder_path and os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    except Exception as e:
        print(f"delete_scan_folder failed: {e}")


# ============================
# General UI helpers
# ============================
def reset_to_home():
    global state, status_text, status_colour
    global current_scan_folder, selected_seedling, run_start_time
    global plate_image_rgb, plate_annotated_rgb, seedling_detections, current_seedling_index

    state = "home"
    selected_seedling = None
    run_start_time = None
    current_scan_folder = None
    plate_image_rgb = None
    plate_annotated_rgb = None
    seedling_detections = []
    current_seedling_index = 0

    if usb_inserted():
        status_text = "USB DETECTED"
        status_colour = GREEN
    else:
        status_text = "READY"
        status_colour = WHITE


def check_screenshot_trigger():
    if os.path.exists(TRIGGER_FILE):
        filename = f"{SCREENSHOT_DIR}/screen_{int(time.time())}.png"
        pygame.image.save(logical_surface, filename)
        os.remove(TRIGGER_FILE)
        print(f"Screenshot saved: {filename}")


def start_focus_camera():
    global picam2, focus_camera_active

    if picam2 is None:
        picam2 = Picamera2()

        config = picam2.create_preview_configuration(
            main={"size": (600, 480), "format": "RGB888"},
            raw={"size": (2050, 1640)},
        )

        picam2.configure(config)
        picam2.start()
        time.sleep(0.3)

    focus_camera_active = True


def stop_focus_camera():
    global picam2, focus_camera_active

    if picam2 is not None:
        try:
            picam2.stop()
        except Exception:
            pass

        try:
            picam2.close()
        except Exception:
            pass

        picam2 = None

    focus_camera_active = False


def set_focus_light(duty_percent):
    if led_pwm is None:
        return

    try:
        led_pwm.ChangeDutyCycle(duty_percent)
    except Exception as e:
        print("Focus light PWM error:", e)


def start_focus_light_timer():
    global focus_light_on, focus_light_off_time
    set_focus_light(FOCUS_LIGHT_DUTY)
    focus_light_on = True
    focus_light_off_time = time.time() + FOCUS_LIGHT_DURATION


def stop_focus_light():
    global focus_light_on, focus_light_off_time
    set_focus_light(0)
    focus_light_on = False
    focus_light_off_time = None


def start_thermal_camera():
    global thermal_cap, thermal_active
    global thermal_last_dtype, thermal_last_min, thermal_last_max

    if thermal_cap is None:
        thermal_cap = cv2.VideoCapture(LEPTON_VIDEO_DEVICE, cv2.CAP_V4L2)

        # Request Y16/16-bit greyscale from PureThermal.
        # Your v4l2-ctl output showed Y16 is available.
        thermal_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        thermal_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))

        if not thermal_cap.isOpened():
            print(f"Failed to open thermal camera: {LEPTON_VIDEO_DEVICE}")
            thermal_cap = None
            thermal_active = False
            return

        try:
            ret, test_frame = thermal_cap.read()
            if ret:
                thermal_last_dtype = str(test_frame.dtype)
                thermal_last_min = int(np.min(test_frame))
                thermal_last_max = int(np.max(test_frame))
                print(
                    "Thermal frame diagnostic:",
                    "dtype=", test_frame.dtype,
                    "shape=", test_frame.shape,
                    "min=", np.min(test_frame),
                    "max=", np.max(test_frame),
                )
        except Exception as e:
            print("Thermal diagnostic failed:", e)

    thermal_active = True


def stop_thermal_camera():
    global thermal_cap, thermal_active

    if thermal_cap is not None:
        try:
            thermal_cap.release()
        except Exception:
            pass

        thermal_cap = None

    thermal_active = False


def read_x_limit():
    try:
        return GPIO.input(X_LIMIT_PIN) == X_LIMIT_PRESSED_STATE
    except Exception as e:
        print("X limit read error:", e)
        return False


def read_z_limit():
    try:
        return GPIO.input(Z_LIMIT_PIN) == Z_LIMIT_PRESSED_STATE
    except Exception as e:
        print("Z limit read error:", e)
        return False


def update_limit_switches():
    global x_limit_pressed, z_limit_pressed
    global prev_x_limit_pressed, prev_z_limit_pressed

    prev_x_limit_pressed = x_limit_pressed
    prev_z_limit_pressed = z_limit_pressed

    x_limit_pressed = read_x_limit()
    z_limit_pressed = read_z_limit()

    if x_limit_pressed != prev_x_limit_pressed:
        print(f"X limit changed: {'PRESSED' if x_limit_pressed else 'RELEASED'}")

    if z_limit_pressed != prev_z_limit_pressed:
        print(f"Z limit changed: {'PRESSED' if z_limit_pressed else 'RELEASED'}")


# ============================================================
# Manual colour / YOLO / radiometric thermal workflow helpers
# ============================================================

def start_guide_video():
    global guide_video_cap
    if guide_video_cap is None and os.path.exists(LOAD_GUIDE_VIDEO_PATH):
        guide_video_cap = cv2.VideoCapture(LOAD_GUIDE_VIDEO_PATH)

def stop_guide_video():
    global guide_video_cap
    if guide_video_cap is not None:
        guide_video_cap.release()
        guide_video_cap = None

def draw_guide_video(surface, rect):
    global guide_video_cap

    if guide_video_cap is None:
        start_guide_video()

    if guide_video_cap is None or not guide_video_cap.isOpened():
        draw_text_center(surface, "GUIDE VIDEO MISSING", font_body, RED, rect.center)
        return

    ret, frame = guide_video_cap.read()

    if not ret:
        guide_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = guide_video_cap.read()

    if not ret:
        draw_text_center(surface, "GUIDE VIDEO FAILED", font_body, RED, rect.center)
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    scale = min(rect.width / w, rect.height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    frame = cv2.resize(frame, (new_w, new_h))
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    x = rect.x + (rect.width - new_w) // 2
    y = rect.y + (rect.height - new_h) // 2

    surface.blit(frame_surface, (x, y))
    pygame.draw.rect(surface, WHITE, (x, y, new_w, new_h), 2)


def load_yolo_model():
    global yolo_model, yolo_model_load_attempted

    if yolo_model_load_attempted:
        return yolo_model

    yolo_model_load_attempted = True

    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"YOLO model not found: {YOLO_MODEL_PATH}")
        return None

    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded: {YOLO_MODEL_PATH}")
        return yolo_model
    except Exception as e:
        print("YOLO model load failed:", e)
        return None


def preprocess_colour_for_yolo(image_rgb):
    if not USE_COLOUR_PREPROCESSING_FOR_YOLO:
        return image_rgb

    try:
        denoised = cv2.fastNlMeansDenoisingColored(image_rgb, None, 4, 4, 7, 21)
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    except Exception as e:
        print("Colour preprocessing failed:", e)
        return image_rgb


def run_seedling_detection(image_rgb):
    model = load_yolo_model()
    detections = []

    if model is None:
        return detections

    try:
        yolo_input = preprocess_colour_for_yolo(image_rgb)

        results = model(
            yolo_input,
            conf=YOLO_CONF_THRESHOLD,
            imgsz=1280,
            iou=0.35,
            verbose=False
        )

        print("YOLO boxes detected:", len(results[0].boxes))

        if not results or results[0].boxes is None:
            return detections

        raw_boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            conf = float(box.conf[0].detach().cpu().numpy()) if box.conf is not None else 0.0
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            raw_boxes.append((x1, y1, x2, y2, conf))

        # Stable numbering:
        # split seedlings into upper/lower rows, then number left-to-right in each row.
        if raw_boxes:
            centers_y = [(b[1] + b[3]) / 2 for b in raw_boxes]
            row_split_y = (min(centers_y) + max(centers_y)) / 2

            top_row = [b for b in raw_boxes if ((b[1] + b[3]) / 2) < row_split_y]
            bottom_row = [b for b in raw_boxes if ((b[1] + b[3]) / 2) >= row_split_y]

            top_row.sort(key=lambda b: b[0])
            bottom_row.sort(key=lambda b: b[0])

            raw_boxes = top_row + bottom_row

        for idx, (x1, y1, x2, y2, conf) in enumerate(raw_boxes, start=1):
            detections.append(
                {
                    "id": idx,
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "thermal_path": None,
                    "thermal_raw_path": None,
                    "temp_c": None,
                    "thermal_source": None,
                }
            )

    except Exception as e:
        print("YOLO detection failed:", e)

    return detections


def draw_colour_image_with_boxes(image_rgb, detections, highlight_id=None):
    global latest_display_boxes
    latest_display_boxes = []

    if image_rgb is None:
        draw_text_center(logical_surface, "NO COLOUR IMAGE", font_body, RED, (SCREEN_W // 2, SCREEN_H // 2))
        return

    info = draw_rgb_image_fit(logical_surface, image_rgb, pygame.Rect(10, 45, 780, 335))
    if info is None:
        return

    x0, y0, new_w, new_h, scale = info

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rect = pygame.Rect(
            int(x0 + x1 * scale),
            int(y0 + y1 * scale),
            int((x2 - x1) * scale),
            int((y2 - y1) * scale),
        )
        latest_display_boxes.append({"id": det["id"], "rect": rect})

        if highlight_id is not None and det["id"] == highlight_id:
            colour = YELLOW
            width = 5
        elif det.get("thermal_path"):
            colour = GREEN
            width = 3
        else:
            colour = BLUE
            width = 3

        pygame.draw.rect(logical_surface, colour, rect, width=width, border_radius=3)
        
        label = str(det["id"])
        if det.get("temp_c") is not None:
            label = f"{det['id']}: {det['temp_c']:.1f}C"

        label_surface = font_small.render(label, True, colour)
        label_x = rect.x
        label_y = rect.bottom + 4

        if label_y + label_surface.get_height() > SCREEN_H - 5:
            label_y = rect.y - label_surface.get_height() - 4

        logical_surface.blit(label_surface, (label_x, label_y))


def renumber_detections():
    # Top row left-to-right, then bottom row left-to-right
    if not seedling_detections:
        return

    centers_y = [(d["bbox"][1] + d["bbox"][3]) / 2 for d in seedling_detections]
    row_split_y = (min(centers_y) + max(centers_y)) / 2

    top = [d for d in seedling_detections if ((d["bbox"][1] + d["bbox"][3]) / 2) < row_split_y]
    bottom = [d for d in seedling_detections if ((d["bbox"][1] + d["bbox"][3]) / 2) >= row_split_y]

    top.sort(key=lambda d: d["bbox"][0])
    bottom.sort(key=lambda d: d["bbox"][0])

    seedling_detections[:] = top + bottom

    for i, det in enumerate(seedling_detections, start=1):
        det["id"] = i


def screen_tap_to_image_xy(x, y):
    if plate_image_rgb is None:
        return None

    img_h, img_w = plate_image_rgb.shape[:2]
    display_rect = pygame.Rect(10, 45, 780, 335)

    scale = min(display_rect.width / img_w, display_rect.height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    x0 = display_rect.x + (display_rect.width - new_w) // 2
    y0 = display_rect.y + (display_rect.height - new_h) // 2

    if not pygame.Rect(x0, y0, new_w, new_h).collidepoint(x, y):
        return None

    img_x = int((x - x0) / scale)
    img_y = int((y - y0) / scale)

    return img_x, img_y


def add_detection_at_screen_tap(x, y):
    pos = screen_tap_to_image_xy(x, y)
    if pos is None:
        return

    img_x, img_y = pos
    img_h, img_w = plate_image_rgb.shape[:2]

    box_w = 60
    box_h = 60

    x1 = max(0, img_x - box_w // 2)
    y1 = max(0, img_y - box_h // 2)
    x2 = min(img_w - 1, img_x + box_w // 2)
    y2 = min(img_h - 1, img_y + box_h // 2)

    seedling_detections.append(
        {
            "id": len(seedling_detections) + 1,
            "bbox": (x1, y1, x2, y2),
            "conf": 1.0,
            "thermal_path": None,
            "thermal_raw_path": None,
            "temp_c": None,
            "thermal_source": "manual_box",
        }
    )

    renumber_detections()


def capture_colour_plate_image():
    global plate_image_rgb, plate_annotated_rgb, seedling_detections
    global status_text, status_colour, review_start_time

    try:
        start_focus_camera()
        set_focus_light(PLATE_CAPTURE_LIGHT_DUTY)
        time.sleep(PLATE_CAPTURE_LIGHT_SECONDS)
        frame = picam2.capture_array()
        stop_focus_light()

        plate_image_rgb = frame.copy()
        seedling_detections = run_seedling_detection(plate_image_rgb)

        '''annotated = plate_image_rgb.copy()
        for det in seedling_detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(
                annotated,
                str(det["id"]),
                (x1, min(y2 + 20, annotated.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
            )
        plate_annotated_rgb = annotated'''

        if current_scan_folder:
            cv2.imwrite(os.path.join(current_scan_folder, "colour_plate_raw.png"), cv2.cvtColor(plate_image_rgb, cv2.COLOR_RGB2BGR))
            # cv2.imwrite(os.path.join(current_scan_folder, "colour_plate_annotated.png"), cv2.cvtColor(plate_annotated_rgb, cv2.COLOR_RGB2BGR))
            if USE_COLOUR_PREPROCESSING_FOR_YOLO:
                preprocessed = preprocess_colour_for_yolo(plate_image_rgb)
                cv2.imwrite(
                    os.path.join(current_scan_folder, "colour_plate_preprocessed_for_yolo.png"),
                    cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR),
                )
            write_metadata()

        status_text = f"{len(seedling_detections)} SEEDLINGS DETECTED"
        status_colour = GREEN if seedling_detections else RED
        review_start_time = time.time()
        return True

    except Exception as e:
        stop_focus_light()
        print("Colour capture failed:", e)
        status_text = "COLOUR CAPTURE FAILED"
        status_colour = RED
        return False


def normalise_lepton_frame(raw_frame):
    if raw_frame is None:
        return None

    frame = raw_frame

    if len(frame.shape) == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]

    if frame.dtype == np.uint16:
        return frame

    if frame.dtype == np.uint8 and len(frame.shape) == 3 and frame.shape[2] == 2:
        try:
            return frame.view(np.uint16).reshape(frame.shape[0], frame.shape[1])
        except Exception:
            pass

    return frame


def lepton_raw_to_celsius(raw_frame):
    frame = normalise_lepton_frame(raw_frame)
    if frame is None:
        return None

    arr = frame.astype(np.float32)

    # 8-bit preview frames cannot provide defensible absolute temperature.
    if np.nanmax(arr) <= 255:
        return None

    temp_c = (arr / LEPTON_RAW_SCALE) - LEPTON_KELVIN_OFFSET
    return THERMAL_CAL_A * temp_c + THERMAL_CAL_B


def analyse_temperature_from_radiometric(temp_c):
    h, w = temp_c.shape[:2]
    roi_w = int(w * THERMAL_CENTRE_ROI_FRACTION)
    roi_h = int(h * THERMAL_CENTRE_ROI_FRACTION)
    x0 = (w - roi_w) // 2
    y0 = (h - roi_h) // 2
    roi = temp_c[y0:y0 + roi_h, x0:x0 + roi_w]

    if roi.size == 0:
        return float(np.nanmean(temp_c))

    if USE_THERMAL_LEAF_MASK:
        threshold = np.nanpercentile(roi, THERMAL_COOL_LEAF_PERCENTILE)
        mask = roi <= threshold
        if np.count_nonzero(mask) >= max(10, int(0.02 * roi.size)):
            return float(np.nanmean(roi[mask]))

    return float(np.nanmean(roi))


def make_thermal_preview_from_frame(frame):
    temp_c = lepton_raw_to_celsius(frame)

    if temp_c is not None:
        if USE_FIXED_THERMAL_DISPLAY_RANGE:
            low = THERMAL_DISPLAY_MIN_C
            high = THERMAL_DISPLAY_MAX_C
        else:
            ambient = float(np.nanmedian(temp_c))
            low = ambient - THERMAL_PREVIEW_RANGE_C
            high = ambient + THERMAL_PREVIEW_RANGE_C
        clipped = np.clip(temp_c, low, high)
        preview = ((clipped - low) / max(0.001, high - low) * 255).astype(np.uint8)
    else:
        normalised = normalise_lepton_frame(frame)
        if normalised is None:
            preview = np.zeros((120, 160), dtype=np.uint8)
        elif len(normalised.shape) == 3:
            preview = cv2.cvtColor(normalised, cv2.COLOR_BGR2GRAY)
        else:
            preview = cv2.normalize(normalised, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    colour_bgr = cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)
    return colour_bgr, temp_c


def get_current_seedling():
    if not seedling_detections:
        return None
    if current_seedling_index < 0 or current_seedling_index >= len(seedling_detections):
        return None
    return seedling_detections[current_seedling_index]


def write_metadata():
    if not current_scan_folder:
        return
    try:
        with open(os.path.join(current_scan_folder, "detections.json"), "w") as f:
            json.dump(seedling_detections, f, indent=2)
    except Exception as e:
        print("Metadata write failed:", e)


def make_annotated_plate_image():
    if plate_image_rgb is None:
        return None

    annotated = plate_image_rgb.copy()

    for det in seedling_detections:
        x1, y1, x2, y2 = det["bbox"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 3)

        label = str(det["id"])
        text_y = min(y2 + 35, annotated.shape[0] - 10)

        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),
            2,
        )

    return annotated


def save_final_colour_image():
    global plate_annotated_rgb

    if current_scan_folder is None or plate_image_rgb is None:
        return

    plate_annotated_rgb = make_annotated_plate_image()

    cv2.imwrite(
        os.path.join(current_scan_folder, "colour_plate_raw.png"),
        cv2.cvtColor(plate_image_rgb, cv2.COLOR_RGB2BGR),
    )

    if plate_annotated_rgb is not None:
        cv2.imwrite(
            os.path.join(current_scan_folder, "colour_plate_annotated.png"),
            cv2.cvtColor(plate_annotated_rgb, cv2.COLOR_RGB2BGR),
        )

    write_metadata()


def write_seedling_temperature_report():
    if not current_scan_folder:
        return

    report_path = os.path.join(current_scan_folder, "seedling_temperatures.txt")
    try:
        with open(report_path, "w") as f:
            f.write("ATLAS seedling thermal results\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("Temperature source: Lepton Y16 radiometric Kelvin x100 where available\n")
            f.write("Leaf extraction: central ROI with cool-pixel mask\n")
            f.write("\n")
            for det in seedling_detections:
                if det.get("temp_c") is None:
                    temp_text = "NO RADIOMETRIC TEMPERATURE AVAILABLE"
                else:
                    temp_text = f"{det['temp_c']:.3f} °C"
                f.write(f"Seedling {det['id']}: {temp_text}\n")
    except Exception as e:
        print("Temperature report write failed:", e)


def capture_current_thermal_image():
    global current_seedling_index, state
    global motor_status_text, motor_status_colour

    det = get_current_seedling()
    if det is None:
        state = "result_overview"
        return

    start_thermal_camera()
    if thermal_cap is None:
        motor_status_text = "NO THERMAL CAM"
        motor_status_colour = RED
        return

    ret, frame = thermal_cap.read()
    if not ret:
        motor_status_text = "NO THERMAL FRAME"
        motor_status_colour = RED
        return

    try:
        preview_bgr, temp_c_frame = make_thermal_preview_from_frame(frame)

        if temp_c_frame is not None:
            det["temp_c"] = analyse_temperature_from_radiometric(temp_c_frame)
            det["thermal_source"] = "radiometric_y16_kelvin_x100"
        else:
            det["temp_c"] = None
            det["thermal_source"] = "preview_only_no_radiometric_temperature"

        if current_scan_folder:
            preview_path = os.path.join(current_scan_folder, f"seedling_{det['id']:02d}_thermal_preview.png")
            preview_bgr = cv2.rotate(preview_bgr, cv2.ROTATE_180)
            cv2.imwrite(preview_path, preview_bgr)
            det["thermal_path"] = preview_path

            raw_path = os.path.join(current_scan_folder, f"seedling_{det['id']:02d}_thermal_raw.npy")
            np.save(raw_path, frame)
            det["thermal_raw_path"] = raw_path

            write_metadata()
            write_seedling_temperature_report()

        current_seedling_index += 1

        if current_seedling_index >= len(seedling_detections):
            stop_thermal_camera()
            state = "result_overview"
        else:
            state = "seedling_colour_guide"

    except Exception as e:
        print("Thermal capture failed:", e)
        motor_status_text = "THERMAL SAVE FAILED"
        motor_status_colour = RED


# ============================================================
# COMMENTED OUT: Stepper HAT B / DRV8825 helper functions
# Keep for future restoration when the replacement HAT arrives.
# ============================================================

"""
def motor_step(step_pin):
    GPIO.output(step_pin, GPIO.HIGH)
    time.sleep(STEP_PULSE_SECONDS)
    GPIO.output(step_pin, GPIO.LOW)
    time.sleep(STEP_GAP_SECONDS)

# Old HAT versions of move_x_mm(), move_z_mm(), and home_motors()
# were intentionally disabled because the active temporary motor system is L298N.
"""


# ============================================================
# RESERVED: L298N stepper motor helper functions
# Used only by Settings > Motor Control.
# ============================================================

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


def set_motor_pins(pins, values):
    for pin, value in zip(pins, values):
        GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)


def release_motor(pins):
    set_motor_pins(pins, [0, 0, 0, 0])


def release_all_motors():
    release_motor(X_MOTOR_PINS)
    release_motor(Z_MOTOR_PINS)


def single_half_step(pins, step_index, direction):
    if direction > 0:
        values = HALF_STEP_SEQUENCE[step_index % len(HALF_STEP_SEQUENCE)]
    else:
        values = list(reversed(HALF_STEP_SEQUENCE))[step_index % len(HALF_STEP_SEQUENCE)]

    set_motor_pins(pins, values)
    time.sleep(STEPPER_DELAY)


def move_motor_steps(axis, steps, direction):
    global motor_busy
    global x_pos_mm, z_pos_mm
    global motor_status_text, motor_status_colour

    if motor_busy:
        motor_status_text = "MOTOR BUSY"
        motor_status_colour = RED
        return False

    if steps <= 0:
        return True

    motor_busy = True

    try:
        if axis == "x":
            pins = X_MOTOR_PINS
        elif axis == "z":
            pins = Z_MOTOR_PINS
        else:
            motor_status_text = "BAD AXIS"
            motor_status_colour = RED
            return False

        for i in range(steps):
            update_limit_switches()

            if axis == "x" and direction < 0 and x_limit_pressed:
                x_pos_mm = 0.0
                motor_status_text = "X LIMIT HIT"
                motor_status_colour = YELLOW
                return False

            if axis == "z" and direction < 0 and z_limit_pressed:
                z_pos_mm = 0.0
                motor_status_text = "Z LIMIT HIT"
                motor_status_colour = YELLOW
                return False

            single_half_step(pins, i, direction)

        return True

    finally:
        release_all_motors()
        motor_busy = False


def move_x_mm(delta_mm):
    global x_pos_mm, motor_status_text, motor_status_colour

    if not motors_homed:
        motor_status_text = "HOME FIRST"
        motor_status_colour = RED
        return

    target = x_pos_mm + delta_mm

    if target < 0:
        target = 0.0
        delta_mm = target - x_pos_mm

    if target > X_MAX_MM:
        motor_status_text = "X MAX"
        motor_status_colour = RED
        return

    if delta_mm == 0:
        return

    direction = 1 if delta_mm > 0 else -1
    steps = int(abs(delta_mm) * X_STEPS_PER_MM)

    ok = move_motor_steps("x", steps, direction)

    if ok:
        x_pos_mm = target
        motor_status_text = f"X {x_pos_mm:.1f}  Z {z_pos_mm:.1f}"
        motor_status_colour = GREEN


def move_z_mm(delta_mm):
    global z_pos_mm, motor_status_text, motor_status_colour

    if not motors_homed:
        motor_status_text = "HOME FIRST"
        motor_status_colour = RED
        return

    target = z_pos_mm + delta_mm

    if target < 0:
        target = 0.0
        delta_mm = target - z_pos_mm

    if target > Z_MAX_MM:
        motor_status_text = "Z MAX"
        motor_status_colour = RED
        return

    if delta_mm == 0:
        return

    direction = 1 if delta_mm > 0 else -1
    steps = int(abs(delta_mm) * Z_STEPS_PER_MM)

    ok = move_motor_steps("z", steps, direction)

    if ok:
        z_pos_mm = target
        motor_status_text = f"X {x_pos_mm:.1f}  Z {z_pos_mm:.1f}"
        motor_status_colour = GREEN


def home_motors():
    global x_pos_mm, z_pos_mm, motors_homed
    global motor_status_text, motor_status_colour

    motor_status_text = "HOMING X..."
    motor_status_colour = YELLOW

    move_motor_steps("x", X_HOME_MAX_STEPS, direction=-1)
    x_pos_mm = 0.0

    time.sleep(0.2)

    motor_status_text = "HOMING Z..."
    motor_status_colour = YELLOW

    move_motor_steps("z", Z_HOME_MAX_STEPS, direction=-1)
    z_pos_mm = 0.0

    motors_homed = True
    motor_status_text = "HOMED"
    motor_status_colour = GREEN


# ============================
# Splash and maintenance mode
# ============================
def show_startup_logo():
    if not SHOW_SPLASH or not os.path.exists(LOGO_PATH):
        return

    try:
        logo = pygame.image.load(LOGO_PATH).convert_alpha()
        logo = pygame.transform.rotate(logo, 180)

        max_w = int(SCREEN_W * 0.82)
        max_h = int(SCREEN_H * 0.82)

        lw, lh = logo.get_size()
        scale = min(max_w / lw, max_h / lh)
        new_size = (int(lw * scale), int(lh * scale))
        logo = pygame.transform.smoothscale(logo, new_size)

        x = (SCREEN_W - logo.get_width()) // 2
        y = (SCREEN_H - logo.get_height()) // 2

        hold_start = time.time()
        while time.time() - hold_start < SPLASH_SECONDS:
            screen.fill(BLACK)
            screen.blit(logo, (x, y))
            pygame.display.flip()
            pygame.time.delay(10)

        fade_steps = 60
        for i in range(fade_steps, -1, -1):
            alpha = int(255 * (i / fade_steps))
            screen.fill(BLACK)
            faded_logo = logo.copy()
            faded_logo.set_alpha(alpha)
            screen.blit(faded_logo, (x, y))
            pygame.display.flip()
            pygame.time.delay(20)

    except Exception as e:
        print("Splash error:", e)


def exit_to_terminal():
    global running
    try:
        with open("/tmp/maintenance_mode", "w") as f:
            f.write("1\n")
    except Exception as e:
        print("Failed to enable maintenance mode:", e)
    running = False


# ============================
# Draw functions
# ============================
def draw_home():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "ATLAS", font_title, WHITE, (SCREEN_W // 2, 80))
    draw_button(logical_surface, start_rect, GREEN, "START", font_button, WHITE)
    draw_text_center(logical_surface, status_text, font_status, status_colour, (SCREEN_W // 2, 350))
    draw_button(logical_surface, settings_rect, DARK_GREY, "SETTINGS", font_small, WHITE)


def draw_settings_menu():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "SETTINGS", font_title, WHITE, (SCREEN_W // 2, 80))
    draw_button(logical_surface, focus_button_rect, BLUE, "FOCUS CAMERA", font_button, BLACK)
    draw_button(logical_surface, thermal_button_rect, RED, "THERMAL VIEW", font_button, BLACK)
    draw_button(logical_surface, motor_button_rect, GREY, "MOTOR CONTROL", font_button, BLACK)
    draw_x_button(logical_surface, settings_close_rect)

    x_text = "X LIMIT: PRESSED" if x_limit_pressed else "X LIMIT: OPEN"
    z_text = "Z LIMIT: PRESSED" if z_limit_pressed else "Z LIMIT: OPEN"
    x_colour = GREEN if x_limit_pressed else WHITE
    z_colour = GREEN if z_limit_pressed else WHITE

    draw_text_center(logical_surface, x_text, font_status, x_colour, (SCREEN_W // 4, 430))
    draw_text_center(logical_surface, z_text, font_status, z_colour, (3 * SCREEN_W // 4, 430))


def draw_focus_camera():
    logical_surface.fill(BLACK)

    if picam2 is not None and focus_camera_active:
        try:
            frame = picam2.capture_array()
            info = draw_rgb_image_fit(logical_surface, frame, pygame.Rect(0, 0, SCREEN_W, SCREEN_H))
            if info is None:
                raise RuntimeError("No frame")
        except Exception as e:
            draw_text_center(logical_surface, "CAMERA PREVIEW FAILED", font_body, RED, (SCREEN_W // 2, SCREEN_H // 2))
            draw_text_center(logical_surface, str(e), font_small, WHITE, (SCREEN_W // 2, SCREEN_H // 2 + 40))
    else:
        draw_text_center(logical_surface, "FOCUS CAMERA", font_title, WHITE, (SCREEN_W // 2, 70))
        draw_text_center(logical_surface, "CAMERA NOT STARTED", font_body, BLUE, (SCREEN_W // 2, SCREEN_H // 2))

    light_button_colour = GREEN if focus_light_on else BLUE
    draw_button(logical_surface, focus_light_rect, light_button_colour, "L", font_small, BLACK)
    draw_x_button(logical_surface, focus_close_rect)


def draw_thermal_frame_to_screen(title_text, capture_buttons=False):
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, title_text, font_small, WHITE, (SCREEN_W // 2, 18))

    if thermal_cap is not None:
        ret, frame = thermal_cap.read()

        if ret:
            preview_bgr, temp_c = make_thermal_preview_from_frame(frame)
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
            preview_rgb = cv2.rotate(preview_rgb, cv2.ROTATE_180)
            draw_rgb_image_fit(logical_surface, preview_rgb, pygame.Rect(10, 35, 780, 370))

            if temp_c is None:
                diag = f"NO RADIOMETRIC TEMP  {thermal_last_dtype} {thermal_last_min}-{thermal_last_max}"
            else:
                diag = f"Y16 OK  ambient {float(np.nanmedian(temp_c)):.1f}C"
            draw_text_center(logical_surface, diag, font_small, WHITE, (SCREEN_W // 2, 370))
        else:
            draw_text_center(logical_surface, "NO FRAME", font_body, RED, (SCREEN_W // 2, SCREEN_H // 2))
    else:
        draw_text_center(logical_surface, "NO THERMAL CAM", font_body, RED, (SCREEN_W // 2, SCREEN_H // 2))

    if capture_buttons:
        draw_button(logical_surface, thermal_back_rect, DARK_GREY, "GUIDE", font_small, WHITE)
        draw_button(logical_surface, thermal_capture_rect, GREEN, "CAPTURE", font_small, WHITE)
    else:
        draw_x_button(logical_surface, focus_close_rect)


def draw_thermal_camera():
    draw_thermal_frame_to_screen("THERMAL CAMERA", capture_buttons=False)


def draw_motor_control():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "MOTOR CONTROL", font_title, WHITE, (SCREEN_W // 2, 55))
    draw_x_button(logical_surface, motor_close_rect)
    draw_button(logical_surface, home_motor_rect, GREEN, "HOME", font_button, WHITE)

    draw_button(logical_surface, x_left_10_rect, RED, "-10", font_button, WHITE)
    draw_button(logical_surface, x_left_1_rect, RED, "-1", font_button, WHITE)
    draw_button(logical_surface, x_right_1_rect, BLUE, "+1", font_button, BLACK)
    draw_button(logical_surface, x_right_10_rect, BLUE, "+10", font_button, BLACK)

    draw_button(logical_surface, z_up_10_rect, BLUE, "+10", font_small, BLACK)
    draw_button(logical_surface, z_up_1_rect, BLUE, "+1", font_small, BLACK)
    draw_button(logical_surface, z_down_1_rect, RED, "-1", font_small, WHITE)
    draw_button(logical_surface, z_down_10_rect, RED, "-10", font_small, WHITE)

    draw_text_center(logical_surface, "X", font_body, WHITE, (SCREEN_W // 2, 180))
    draw_text_center(logical_surface, "Z", font_body, WHITE, (SCREEN_W // 2, 270))

    draw_text_center(
        logical_surface,
        motor_status_text,
        font_status,
        motor_status_colour,
        (SCREEN_W // 2, 455),
    )


def draw_confirm_seedlings():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "SEEDLINGS LOADED?", font_title, WHITE, (SCREEN_W // 2, 110))
    draw_button(logical_surface, yes_rect, GREEN, "YES", font_button, WHITE)
    draw_button(logical_surface, no_rect, RED, "NO", font_button, WHITE)


def draw_colour_live():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "ALIGN PLATE", font_body, WHITE, (SCREEN_W // 2, 40))

    if picam2 is not None and focus_camera_active:
        try:
            frame = picam2.capture_array()
            draw_rgb_image_fit(logical_surface, frame, colour_live_view_rect)
        except Exception as e:
            draw_text_center(logical_surface, "LIVE COLOUR FEED FAILED", font_body, RED, (SCREEN_W // 2, SCREEN_H // 2))
            draw_text_center(logical_surface, str(e), font_small, WHITE, (SCREEN_W // 2, SCREEN_H // 2 + 40))
    else:
        draw_text_center(logical_surface, "COLOUR CAMERA NOT STARTED", font_body, RED, (SCREEN_W // 2, SCREEN_H // 2))

    draw_image_button(logical_surface, colour_capture_rect, GREEN, CAMERA_LOGO_PATH)


def draw_colour_review():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "CONFIRM DETECTION", font_body, WHITE, (SCREEN_W // 2, 25))

    draw_colour_image_with_boxes(plate_image_rgb, seedling_detections)

    draw_text_center(
        logical_surface,
        f"Detected seedlings: {len(seedling_detections)}",
        font_small,
        WHITE,
        (SCREEN_W // 2, 345),
    )

    remaining = REVIEW_AUTO_SECONDS
    if review_start_time is not None:
        remaining = max(0, REVIEW_AUTO_SECONDS - (time.time() - review_start_time))

    bar_outer = pygame.Rect(240, 362, 320, 12)
    pygame.draw.rect(logical_surface, GREY, bar_outer, width=2, border_radius=6)

    progress = remaining / REVIEW_AUTO_SECONDS
    bar_inner = pygame.Rect(243, 365, int(314 * progress), 6)
    pygame.draw.rect(logical_surface, YELLOW, bar_inner, border_radius=4)

    draw_text_center(
        logical_surface,
        f"Auto continue in {remaining:.1f}s",
        font_small,
        YELLOW,
        (SCREEN_W // 2, 380),
    )

    draw_button(logical_surface, colour_retake_rect, RED, "RETAKE", font_small, WHITE)
    draw_button(logical_surface, edit_boxes_rect, BLUE, "EDIT", font_small, WHITE)
    draw_button(logical_surface, colour_confirm_rect, GREEN, "CONFIRM", font_small, WHITE)


def draw_box_edit():
    logical_surface.fill(BLACK)

    draw_text_center(logical_surface, "EDIT SEEDLINGS", font_body, WHITE, (SCREEN_W // 2, 25))
    draw_colour_image_with_boxes(plate_image_rgb, seedling_detections)

    draw_text_center(
        logical_surface,
        "Tap box to remove.",
        font_small,
        WHITE,
        (SCREEN_W // 2, 410),
    )

    draw_text_center(
        logical_surface,
        "Tap empty space to add.",
        font_small,
        WHITE,
        (SCREEN_W // 2, 440),
    )

    draw_button(logical_surface, edit_clear_rect, RED, "CLEAR", font_small, WHITE)
    draw_button(logical_surface, edit_done_rect, GREEN, "DONE", font_small, WHITE)


def draw_manual_position():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "MANUAL THERMAL STAGE", font_body, WHITE, (SCREEN_W // 2, 80))
    draw_text_center(logical_surface, "Move the plate into the chamber.", font_body, WHITE, (SCREEN_W // 2, 160))
    draw_text_center(logical_surface, "Position seedling 1 in front of the thermal camera.", font_small, BLUE, (SCREEN_W // 2, 215))
    draw_button(logical_surface, manual_ready_rect, GREEN, "READY", font_button, WHITE)


def draw_seedling_colour_guide():
    logical_surface.fill(BLACK)
    det = get_current_seedling()
    if det is None:
        draw_text_center(logical_surface, "NO MORE SEEDLINGS", font_body, GREEN, (SCREEN_W // 2, SCREEN_H // 2))
        return

    draw_text_center(logical_surface, f"POSITION SEEDLING {det['id']}", font_body, WHITE, (SCREEN_W // 2, 20))
    draw_colour_image_with_boxes(plate_image_rgb, seedling_detections, highlight_id=det["id"])
    draw_text_center(logical_surface, "Align seedling in chamber", font_small, WHITE, (SCREEN_W // 2, 430))
    draw_button(logical_surface, guide_back_rect, DARK_GREY, "BACK", font_small, WHITE)
    draw_button(logical_surface, guide_thermal_rect, BLUE, "THERMAL", font_small, BLACK)


def draw_seedling_thermal():
    det = get_current_seedling()
    title = "THERMAL CAPTURE" if det is None else f"SEEDLING {det['id']} THERMAL"
    draw_thermal_frame_to_screen(title, capture_buttons=True)


def draw_load_plate_help():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "INSERT PETRI DISH INTO STAGE", font_body, WHITE, (SCREEN_W // 2, 40))
    draw_button(logical_surface, help_close_rect, RED, "X", font_button, WHITE)

    guide_rect = pygame.Rect(SCREEN_W // 2 - 336, 80, 672, 378)
    draw_guide_video(logical_surface, guide_rect)


def draw_running_placeholder():
    logical_surface.fill(BLACK)
    draw_button(logical_surface, stop_small_rect, RED, "STOP", font_small, WHITE)
    draw_text_center(logical_surface, "RUNNING SCAN", font_title, WHITE, (SCREEN_W // 2, 120))

    if run_start_time is not None:
        elapsed = time.time() - run_start_time
        remaining = max(0, int(10 - elapsed))
        draw_text_center(logical_surface, f"PLACEHOLDER RUN: {remaining}s", font_body, BLUE, (SCREEN_W // 2, 220))

    bar_outer = pygame.Rect(140, 280, 520, 32)
    pygame.draw.rect(logical_surface, GREY, bar_outer, width=2, border_radius=12)

    if run_start_time is not None:
        progress = min(1.0, (time.time() - run_start_time) / 10.0)
        bar_inner = pygame.Rect(144, 284, int(512 * progress), 24)
        pygame.draw.rect(logical_surface, GREEN, bar_inner, border_radius=10)


def draw_result_overview():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "RESULTS OVERVIEW", font_title, WHITE, (SCREEN_W // 2, 40))

    if plate_image_rgb is not None and seedling_detections:
        draw_colour_image_with_boxes(plate_image_rgb, seedling_detections)
    else:
        image_rect = pygame.Rect(40, 110, 600, 360)
        pygame.draw.rect(logical_surface, DARK_GREY, image_rect, border_radius=14)
        for item in fake_seedlings:
            r = item["rect"]
            pygame.draw.rect(logical_surface, YELLOW, r, width=3, border_radius=10)
            draw_text_center(logical_surface, f"S{item['id']}  {item['temp']:.1f}C", font_small, WHITE, r.center)

    draw_button(logical_surface, result_next_rect, BLUE, "NEXT", font_small, BLACK)


def draw_seedling_detail():
    logical_surface.fill(BLACK)
    draw_button(logical_surface, detail_close_rect, RED, "X", font_button, WHITE)

    if selected_seedling is None:
        draw_text_center(logical_surface, "NO SEEDLING SELECTED", font_body, WHITE, (SCREEN_W // 2, SCREEN_H // 2))
        return

    draw_text_center(logical_surface, f"SEEDLING {selected_seedling['id']}", font_title, WHITE, (SCREEN_W // 2, 60))

    therm_rect = pygame.Rect(180, 110, 440, 230)
    pygame.draw.rect(logical_surface, DARK_GREY, therm_rect, border_radius=16)

    thermal_path = selected_seedling.get("thermal_path") if isinstance(selected_seedling, dict) else None
    if thermal_path and os.path.exists(thermal_path):
        try:
            img = cv2.imread(thermal_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (therm_rect.width, therm_rect.height))
            surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            logical_surface.blit(surface, therm_rect.topleft)
        except Exception:
            draw_text_center(logical_surface, "THERMAL IMAGE LOAD FAILED", font_small, RED, therm_rect.center)
    else:
        draw_text_center(logical_surface, "LEPTON IMAGE PLACEHOLDER", font_body, RED, therm_rect.center)

    if isinstance(selected_seedling, dict) and "temp_c" in selected_seedling:
        temp_c = selected_seedling.get("temp_c")
        temp_text = "TEMP: NO RADIOMETRIC DATA" if temp_c is None else f"TEMP: {temp_c:.2f}C"
    else:
        temp_text = f"TEMP: {selected_seedling['temp']:.1f}C"

    draw_text_center(logical_surface, temp_text, font_status, WHITE, (SCREEN_W // 2, 390))


def draw_save_decision():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "SAVE SCAN DATA?", font_title, WHITE, (SCREEN_W // 2, 90))
    draw_button(logical_surface, discard_rect, RED, "DISCARD & RETURN", font_small, WHITE)
    draw_button(logical_surface, save_rect, GREEN, "SAVE DATA", font_small, WHITE)

    if current_scan_folder:
        folder_name = os.path.basename(current_scan_folder)
        draw_text_center(logical_surface, folder_name, font_small, BLUE, (SCREEN_W // 2, 260))


def draw_save_complete():
    logical_surface.fill(BLACK)
    draw_text_center(logical_surface, "DATA SAVED", font_title, GREEN, (SCREEN_W // 2, 120))
    if current_scan_folder:
        draw_text_center(logical_surface, os.path.basename(current_scan_folder), font_status, WHITE, (SCREEN_W // 2, 220))
    draw_button(logical_surface, save_done_rect, BLUE, "RETURN TO START", font_button, BLACK)


# ============================
# Input handling
# ============================
def handle_home_press(x, y):
    global state, status_text, status_colour

    if start_rect.collidepoint(x, y):
        if not usb_inserted():
            status_text = "PLEASE INSERT USB DRIVE"
            status_colour = RED
        else:
            if mount_usb():
                state = "confirm_seedlings"
            else:
                status_text = "USB MOUNT FAILED"
                status_colour = RED

    elif settings_rect.collidepoint(x, y):
        state = "settings_menu"


def handle_settings_menu_press(x, y):
    global state

    if settings_close_rect.collidepoint(x, y):
        state = "home"

    elif focus_button_rect.collidepoint(x, y):
        start_focus_camera()
        state = "focus_camera"

    elif thermal_button_rect.collidepoint(x, y):
        start_thermal_camera()
        state = "thermal_camera"

    elif motor_button_rect.collidepoint(x, y):
        state = "motor_control"


def handle_focus_camera_press(x, y):
    global state

    if focus_close_rect.collidepoint(x, y):
        stop_focus_light()
        stop_focus_camera()
        state = "settings_menu"

    elif focus_light_rect.collidepoint(x, y):
        start_focus_light_timer()


def handle_thermal_camera_press(x, y):
    global state

    if focus_close_rect.collidepoint(x, y):
        stop_thermal_camera()
        state = "settings_menu"


def handle_motor_control_press(x, y):
    global state, motor_status_text, motor_status_colour

    if motor_close_rect.collidepoint(x, y):
        release_all_motors()
        motor_status_text = "MOTORS RELEASED"
        motor_status_colour = GREEN
        state = "settings_menu"

    elif home_motor_rect.collidepoint(x, y):
        home_motors()

    elif x_left_10_rect.collidepoint(x, y):
        move_x_mm(-10)

    elif x_left_1_rect.collidepoint(x, y):
        move_x_mm(-1)

    elif x_right_1_rect.collidepoint(x, y):
        move_x_mm(1)

    elif x_right_10_rect.collidepoint(x, y):
        move_x_mm(10)

    elif z_up_10_rect.collidepoint(x, y):
        move_z_mm(10)

    elif z_up_1_rect.collidepoint(x, y):
        move_z_mm(1)

    elif z_down_1_rect.collidepoint(x, y):
        move_z_mm(-1)

    elif z_down_10_rect.collidepoint(x, y):
        move_z_mm(-10)


def handle_confirm_press(x, y):
    global state, current_scan_folder, status_text, status_colour

    if yes_rect.collidepoint(x, y):
        current_scan_folder = create_scan_folder()
        if current_scan_folder is None:
            unmount_usb()
            state = "home"
            status_text = "USB SAVE FOLDER FAILED"
            status_colour = RED
            return

        start_focus_camera()
        state = "colour_live"

    elif no_rect.collidepoint(x, y):
        state = "load_plate_help"


def handle_colour_live_press(x, y):
    global state
    if colour_capture_rect.collidepoint(x, y):
        if capture_colour_plate_image():
            state = "colour_review"


def handle_colour_review_press(x, y):
    global state, current_seedling_index, review_start_time

    if colour_retake_rect.collidepoint(x, y):
        review_start_time = None
        state = "colour_live"

    elif edit_boxes_rect.collidepoint(x, y):
        review_start_time = None
        state = "box_edit"

    elif colour_confirm_rect.collidepoint(x, y):
        review_start_time = None
        stop_focus_camera()
        renumber_detections()
        save_final_colour_image()
        current_seedling_index = 0
        state = "manual_position" if seedling_detections else "result_overview"


def handle_box_edit_press(x, y):
    global state, current_seedling_index

    if edit_done_rect.collidepoint(x, y):
        renumber_detections()
        save_final_colour_image()
        current_seedling_index = 0
        state = "manual_position" if seedling_detections else "result_overview"
        return

    if edit_clear_rect.collidepoint(x, y):
        seedling_detections.clear()
        return

    for item in latest_display_boxes:
        if item["rect"].collidepoint(x, y):
            seedling_detections[:] = [
                d for d in seedling_detections if d["id"] != item["id"]
            ]
            renumber_detections()
            return

    add_detection_at_screen_tap(x, y)


def handle_manual_position_press(x, y):
    global state
    if manual_ready_rect.collidepoint(x, y):
        state = "seedling_colour_guide"


def handle_seedling_colour_guide_press(x, y):
    global state
    if guide_back_rect.collidepoint(x, y):
        state = "manual_position"
    elif guide_thermal_rect.collidepoint(x, y):
        start_thermal_camera()
        state = "seedling_thermal"
    else:
        det = get_current_seedling()
        if det is not None:
            for item in latest_display_boxes:
                if item["id"] == det["id"] and item["rect"].collidepoint(x, y):
                    start_thermal_camera()
                    state = "seedling_thermal"
                    return


def handle_seedling_thermal_press(x, y):
    global state
    if thermal_back_rect.collidepoint(x, y):
        state = "seedling_colour_guide"
    elif thermal_capture_rect.collidepoint(x, y):
        capture_current_thermal_image()


def handle_help_press(x, y):
    global state
    if help_close_rect.collidepoint(x, y):
        stop_guide_video()
        state = "confirm_seedlings"


def handle_running_press(x, y):
    global current_scan_folder, run_start_time
    if stop_small_rect.collidepoint(x, y):
        delete_scan_folder(current_scan_folder)
        current_scan_folder = None
        run_start_time = None
        unmount_usb()
        reset_to_home()


def handle_result_press(x, y):
    global state, selected_seedling

    if seedling_detections:
        for item in latest_display_boxes:
            if item["rect"].collidepoint(x, y):
                selected_seedling = next((d for d in seedling_detections if d["id"] == item["id"]), None)
                state = "seedling_detail"
                return
    else:
        for item in fake_seedlings:
            if item["rect"].collidepoint(x, y):
                selected_seedling = item
                state = "seedling_detail"
                return

    if result_next_rect.collidepoint(x, y):
        state = "save_decision"


def handle_detail_press(x, y):
    global state
    if detail_close_rect.collidepoint(x, y):
        state = "result_overview"


def handle_save_decision_press(x, y):
    global state, current_scan_folder

    if discard_rect.collidepoint(x, y):
        delete_scan_folder(current_scan_folder)
        current_scan_folder = None
        unmount_usb()
        reset_to_home()

    elif save_rect.collidepoint(x, y):
        write_metadata()
        write_seedling_temperature_report()
        state = "save_complete"


def handle_save_complete_press(x, y):
    if save_done_rect.collidepoint(x, y):
        unmount_usb()
        reset_to_home()


# ============================
# Start background touch thread
# ============================
thread = threading.Thread(target=touch_thread, daemon=True)
thread.start()

show_startup_logo()
reset_to_home()

prev_touch_pressed = False
clock = pygame.time.Clock()


# ============================
# Main loop
# ============================
while running:
    if state == "home":
        if usb_inserted():
            if status_text in ("READY", "PLEASE INSERT USB DRIVE"):
                status_text = "USB DETECTED"
                status_colour = GREEN
        else:
            if status_text == "USB DETECTED":
                status_text = "READY"
                status_colour = WHITE

    if focus_light_on and focus_light_off_time is not None:
        if time.time() >= focus_light_off_time:
            stop_focus_light()

    if state == "colour_review" and review_start_time is not None:
        if time.time() - review_start_time >= REVIEW_AUTO_SECONDS:
            review_start_time = None
            stop_focus_camera()
            renumber_detections()
            save_final_colour_image()
            current_seedling_index = 0
            state = "manual_position" if seedling_detections else "result_overview"

    update_limit_switches()

    if state == "home":
        draw_home()
    elif state == "confirm_seedlings":
        draw_confirm_seedlings()
    elif state == "colour_live":
        draw_colour_live()
    elif state == "colour_review":
        draw_colour_review()
    elif state == "box_edit":
        draw_box_edit()
    elif state == "manual_position":
        draw_manual_position()
    elif state == "seedling_colour_guide":
        draw_seedling_colour_guide()
    elif state == "seedling_thermal":
        draw_seedling_thermal()
    elif state == "load_plate_help":
        draw_load_plate_help()
    elif state == "running_placeholder":
        draw_running_placeholder()
    elif state == "result_overview":
        draw_result_overview()
    elif state == "seedling_detail":
        draw_seedling_detail()
    elif state == "save_decision":
        draw_save_decision()
    elif state == "save_complete":
        draw_save_complete()
    elif state == "settings_menu":
        draw_settings_menu()
    elif state == "focus_camera":
        draw_focus_camera()
    elif state == "thermal_camera":
        draw_thermal_camera()
    elif state == "motor_control":
        draw_motor_control()

    # Hidden maintenance exit (home screen only, press-and-hold)
    if state == "home":
        if touch_pressed and touch_x is not None and touch_y is not None:
            if hidden_exit_rect.collidepoint(touch_x, touch_y):
                if hidden_exit_hold_start is None:
                    hidden_exit_hold_start = time.time()
                elif time.time() - hidden_exit_hold_start >= HIDDEN_EXIT_HOLD_SECONDS:
                    exit_to_terminal()
            else:
                hidden_exit_hold_start = None
        else:
            hidden_exit_hold_start = None
    else:
        hidden_exit_hold_start = None

    # Touch edge trigger
    if touch_pressed and not prev_touch_pressed and touch_x is not None and touch_y is not None:
        if DEBUG_TOUCH:
            print(f"Touch at: {touch_x}, {touch_y}, state={state}")

        if state == "home":
            handle_home_press(touch_x, touch_y)
        elif state == "confirm_seedlings":
            handle_confirm_press(touch_x, touch_y)
        elif state == "colour_live":
            handle_colour_live_press(touch_x, touch_y)
        elif state == "colour_review":
            handle_colour_review_press(touch_x, touch_y)
        elif state == "manual_position":
            handle_manual_position_press(touch_x, touch_y)
        elif state == "seedling_colour_guide":
            handle_seedling_colour_guide_press(touch_x, touch_y)
        elif state == "seedling_thermal":
            handle_seedling_thermal_press(touch_x, touch_y)
        elif state == "load_plate_help":
            handle_help_press(touch_x, touch_y)
        elif state == "running_placeholder":
            handle_running_press(touch_x, touch_y)
        elif state == "box_edit":
            handle_box_edit_press(touch_x, touch_y)
        elif state == "result_overview":
            handle_result_press(touch_x, touch_y)
        elif state == "seedling_detail":
            handle_detail_press(touch_x, touch_y)
        elif state == "save_decision":
            handle_save_decision_press(touch_x, touch_y)
        elif state == "save_complete":
            handle_save_complete_press(touch_x, touch_y)
        elif state == "settings_menu":
            handle_settings_menu_press(touch_x, touch_y)
        elif state == "focus_camera":
            handle_focus_camera_press(touch_x, touch_y)
        elif state == "thermal_camera":
            handle_thermal_camera_press(touch_x, touch_y)
        elif state == "motor_control":
            handle_motor_control_press(touch_x, touch_y)

    prev_touch_pressed = touch_pressed

    rotated = pygame.transform.rotate(logical_surface, 180)
    screen.blit(rotated, (0, 0))

    check_screenshot_trigger()

    pygame.display.flip()
    clock.tick(60)


# ============================
# Cleanup
# ============================
try:
    release_all_motors()
except Exception:
    pass

try:
    stop_thermal_camera()
except Exception:
    pass

try:
    stop_focus_light()
except Exception:
    pass

try:
    stop_focus_camera()
except Exception:
    pass

try:
    if led_pwm is not None:
        led_pwm.stop()
except Exception:
    pass

try:
    stop_guide_video()
except Exception:
    pass

# ============================================================
# COMMENTED OUT: Stepper HAT cleanup
# ============================================================
# try:
#     GPIO.output(X_ENABLE_PIN, GPIO.HIGH)
#     GPIO.output(Z_ENABLE_PIN, GPIO.HIGH)
# except Exception:
#     pass

try:
    GPIO.cleanup()
except Exception:
    pass

pygame.quit()
