"""ASTRA Optical Vision Sensor (Webcam Hardware Capture & Audio Shutter).

Enables autonomous computer vision for ASTRA via Linux V4L2 webcams (e.g. Logitech C925e)
and macOS AVFoundation cameras (e.g. MacBook built-in FaceTime HD camera),
with high-speed 1080p frame grabbing and audible mechanical shutter feedback.
"""
import glob
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

SYSTEM_OS = platform.system()  # 'Linux', 'Darwin', 'Windows'

# Sound effect candidates
LINUX_SHUTTER_SOUNDS = [
    "/usr/share/sounds/freedesktop/stereo/camera-shutter.oga",
    "/usr/share/sounds/freedesktop/stereo/screen-capture.oga",
]

MACOS_SHUTTER_SOUNDS = [
    "/System/Library/Components/CoreAudio.component/Contents/SharedSupport/SystemSounds/system/Screen Capture.aif",
    "/System/Library/Sounds/Tink.aiff",
    "/System/Library/Sounds/Pop.aiff",
]


def get_camera_backend() -> str:
    """Return description of active camera backend and platform."""
    if SYSTEM_OS == "Darwin":
        return "macOS AVFoundation (Built-in / USB)"
    elif SYSTEM_OS == "Linux":
        dev = get_camera_device()
        return f"Linux Video4Linux2 ({dev})"
    elif SYSTEM_OS == "Windows":
        return "Windows DirectShow"
    return f"Generic ({SYSTEM_OS})"


def get_camera_device(preferred: str | None = None) -> str:
    """Resolve active camera device node."""
    if SYSTEM_OS == "Darwin":
        # macOS uses avfoundation device strings or indexes, default to "default"
        return preferred if preferred else "default"

    if preferred and os.path.exists(preferred):
        return preferred

    env_dev = os.environ.get("ASTRA_CAMERA")
    if env_dev and os.path.exists(env_dev):
        return env_dev

    # Check Linux /dev/video0 first, then glob for others
    candidates = ["/dev/video0", "/dev/video1", "/dev/video2"]
    for dev in candidates:
        if os.path.exists(dev):
            return dev

    globbed = sorted(glob.glob("/dev/video*"))
    if globbed:
        return globbed[0]

    return "/dev/video0"


def play_shutter_sound():
    """Play instantaneous camera shutter audio cue in the background."""
    if SYSTEM_OS == "Darwin":
        sound_path = None
        for snd in MACOS_SHUTTER_SOUNDS:
            if os.path.isfile(snd):
                sound_path = snd
                break
        if sound_path and shutil.which("afplay"):
            try:
                subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
        return

    # Linux sound playback
    sound_path = None
    for snd in LINUX_SHUTTER_SOUNDS:
        if os.path.isfile(snd):
            sound_path = snd
            break

    if not sound_path:
        return

    for player in ["paplay", "pw-play", "canberra-gtk-play", "aplay"]:
        if shutil.which(player):
            try:
                if player == "canberra-gtk-play":
                    subprocess.Popen([player, "-i", "camera-shutter"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.Popen([player, sound_path],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass


def notify_camera_signal(image_path: str, query: str = "", status: str = "analyzing", signals_dir: str = "/dev/shm/signals"):
    """Write camera event to signal bus for Steampunk Viewfinder and HUDs."""
    try:
        # Fallback to /tmp/signals if /dev/shm is unavailable (e.g. macOS)
        if not os.path.exists("/dev/shm") and signals_dir.startswith("/dev/shm"):
            signals_dir = "/tmp/signals"
        os.makedirs(signals_dir, exist_ok=True)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "query": query,
            "status": status,
            "backend": get_camera_backend(),
            "device": get_camera_device()
        }
        sig_file = os.path.join(signals_dir, ".camera_snap")
        with open(sig_file, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def snap_webcam_frame(
    output_dir: str | None = None,
    device: str | None = None,
    width: int = 1920,
    height: int = 1080,
    query: str = ""
) -> str | None:
    """Capture a single crisp frame from the webcam at 1080p.

    Supports Linux V4L2 and macOS AVFoundation.
    Returns the path to the newly captured image file, or None on failure.
    """
    if not output_dir:
        for candidate_dir in ["/workspaces_nvme/astra_pic", os.path.expanduser("~/Pictures/astra_eyes")]:
            try:
                os.makedirs(candidate_dir, exist_ok=True)
                output_dir = candidate_dir
                break
            except Exception:
                continue

    if not output_dir:
        output_dir = "/tmp"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"snap_{timestamp}.jpg")

    # 1. Play audible shutter cue immediately
    play_shutter_sound()

    # 2. Capture based on operating system
    if SYSTEM_OS == "Darwin":
        # macOS AVFoundation capture pipeline
        dev_target = device if device and device != "/dev/video0" else "default"
        cmd = [
            "ffmpeg", "-y",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", f"{width}x{height}",
            "-i", dev_target,
            "-frames:v", "1",
            "-update", "1",
            out_file
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=7)
            if res.returncode == 0 and os.path.isfile(out_file) and os.path.getsize(out_file) > 1000:
                notify_camera_signal(out_file, query=query, status="analyzing")
                print(f" [ASTRA] 📸 Frame snapped: {out_file} (macOS AVFoundation)")
                return out_file
        except Exception as e:
            print(f" [ASTRA] ⚠️ macOS AVFoundation ffmpeg error: {e}")

        # Fallback for macOS: imagesnap if installed via homebrew
        if shutil.which("imagesnap"):
            try:
                res = subprocess.run(["imagesnap", "-w", "0.8", out_file], capture_output=True, timeout=6)
                if res.returncode == 0 and os.path.isfile(out_file) and os.path.getsize(out_file) > 1000:
                    notify_camera_signal(out_file, query=query, status="analyzing")
                    print(f" [ASTRA] 📸 Frame snapped: {out_file} (imagesnap)")
                    return out_file
            except Exception:
                pass

        return None

    # Linux Video4Linux2 capture pipeline
    target_device = get_camera_device(device)
    if not os.path.exists(target_device):
        print(f" [ASTRA] ⚠️ Camera device {target_device} not found.")
        return None

    # Hardware MJPEG fast capture
    cmd = [
        "ffmpeg", "-y",
        "-f", "v4l2",
        "-input_format", "mjpeg",
        "-video_size", f"{width}x{height}",
        "-i", target_device,
        "-frames:v", "1",
        "-update", "1",
        out_file
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, timeout=6)
        if res.returncode == 0 and os.path.isfile(out_file) and os.path.getsize(out_file) > 1000:
            notify_camera_signal(out_file, query=query, status="analyzing")
            print(f" [ASTRA] 📸 Frame snapped: {out_file} ({width}x{height} MJPEG)")
            return out_file
    except subprocess.TimeoutExpired:
        print(f" [ASTRA] ⚠️ ffmpeg MJPEG capture timed out on {target_device}")
    except Exception as e:
        print(f" [ASTRA] ⚠️ Primary capture error: {e}")

    # Fallback: standard YUYV / auto capture format at 1280x720
    fallback_cmd = [
        "ffmpeg", "-y",
        "-f", "v4l2",
        "-video_size", "1280x720",
        "-i", target_device,
        "-frames:v", "1",
        "-update", "1",
        out_file
    ]
    try:
        res = subprocess.run(fallback_cmd, capture_output=True, timeout=6)
        if res.returncode == 0 and os.path.isfile(out_file) and os.path.getsize(out_file) > 1000:
            notify_camera_signal(out_file, query=query, status="analyzing")
            print(f" [ASTRA] 📸 Fallback frame snapped: {out_file} (1280x720)")
            return out_file
    except Exception as ex:
        print(f" [ASTRA] ❌ Camera capture fallback failed: {ex}")

    return None
