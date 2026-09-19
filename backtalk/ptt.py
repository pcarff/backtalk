# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hold-to-talk — a global key listener.

HOLD the key -> mic opens. RELEASE -> mic closes and the utterance is
processed. The button IS the voice-activity detector, which is why this
mode is speaker-safe with no headphones: the mic simply isn't open while
the assistant talks, unless you press the key — and pressing while it
talks interrupts it.

THE KEY-REPEAT TRAP (the bug that kills every naive build): the OS fires
on_press events CONTINUOUSLY while a key is held. Without the held-state
filter below, every repeat reads as a fresh press and keeps cancelling
the reply before it can speak.

AND THE HALF THAT TRAP HIDES: some keyboards send auto-repeat as full
DOWN/UP PAIRS rather than the repeated DOWN-only stream. Filtering the
presses and trusting every release then breaks the OTHER way -- a single
hold is chopped into dozens of ~50ms recordings, each too short to
transcribe, and the whole thing is SILENT. No exception, no log line,
nothing to search for; it simply reads as "the microphone does not work".
Measured in the field on a Logitech MX Mechanical through a Bolt
receiver: one 2.6-second hold produced 186 key events and about fifty
recordings. So a release is never trusted on sight -- see is_held().

macOS needs Input Monitoring permission for the hosting terminal
(System Settings -> Privacy & Security -> Input Monitoring). Windows
works out of the box; some Linux desktops need the user in the `input`
group or an X11 session.
"""
import threading
import time

from pynput import keyboard


def resolve_key(name: str):
    """'home' / 'f13' / 'right_alt' / any single character -> pynput key."""
    name = (name or "home").strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    # Friendly names -> pynput's names. pynput calls the right option key
    # alt_r, not right_alt; the docs speak human, this map translates.
    # (Field-caught: right_alt silently fell back to home, which Mac
    # laptops cannot press, so the voice looked healthy and never fired.)
    aliases = {
        "right_alt": "alt_r", "left_alt": "alt_l",
        "right_option": "alt_r", "left_option": "alt_l",
        "right_ctrl": "ctrl_r", "left_ctrl": "ctrl_l",
        "right_cmd": "cmd_r", "left_cmd": "cmd_l",
        "right_shift": "shift_r", "left_shift": "shift_l",
    }
    name = aliases.get(name, name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        print(f"[ptt] unknown key {name!r} — falling back to 'home'",
              flush=True)
        return keyboard.Key.home


class PTTListener:
    # How long a release must stand unchallenged before it is believed.
    # Comfortably longer than any keyboard's auto-repeat period / debounce
    # and short enough that letting go still feels natural.
    RELEASE_GRACE = 0.25

    def __init__(self, key="home"):
        self._key = resolve_key(key) if isinstance(key, str) else key
        self._held = False
        self._press_t = None
        self._release_t = None          # a release awaiting confirmation
        self._press_evt = threading.Event()
        self._listener = keyboard.Listener(on_press=self._on_press,
                                           on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _matches(self, k) -> bool:
        if k == self._key:
            return True
        # On Linux X11, Right Alt can report as alt_r or alt_gr
        if self._key in (keyboard.Key.alt_r, keyboard.Key.alt_gr):
            if k in (keyboard.Key.alt_r, keyboard.Key.alt_gr):
                return True
        # Left Alt can report as alt_l or alt
        if self._key in (keyboard.Key.alt_l, keyboard.Key.alt):
            if k in (keyboard.Key.alt_l, keyboard.Key.alt):
                return True
        # Ctrl aliases
        if self._key in (keyboard.Key.ctrl_r, keyboard.Key.ctrl_l, keyboard.Key.ctrl):
            if k in (keyboard.Key.ctrl_r, keyboard.Key.ctrl_l, keyboard.Key.ctrl):
                return True
        # KeyCode comparison by vk or char
        if isinstance(self._key, keyboard.KeyCode) and isinstance(k, keyboard.KeyCode):
            if self._key.vk is not None and self._key.vk == k.vk:
                return True
            if self._key.char is not None and self._key.char == k.char:
                return True
        return False

    def _on_press(self, k):
        if not self._matches(k):
            return
        # A press cancels any pending release: that release was auto-repeat,
        # not a human letting go.
        self._release_t = None
        if not self._held:                      # filter key-repeat
            self._held = True
            self._press_t = time.monotonic()
            self._press_evt.set()

    def _on_release(self, k):
        if self._matches(k):
            # PROVISIONAL. Believed only if no press follows; see _settle().
            self._release_t = time.monotonic()

    def _settle(self):
        """Commit a release that has stood unchallenged for the grace window."""
        r = self._release_t
        if self._held and r is not None and \
                time.monotonic() - r >= self.RELEASE_GRACE:
            self._held = False
            self._release_t = None
            self._press_t = None
        elif self._held and self._press_t is not None and \
                time.monotonic() - self._press_t >= 45.0:
            # Watchdog: release key if held longer than 45s (e.g. lost X11 KeyRelease event)
            self._held = False
            self._release_t = None
            self._press_t = None

    def wait_press(self):
        """Block until the key goes DOWN (one event per physical press)."""
        # Settled on a loop, not once. A release landing after the last
        # is_held() poll leaves _held provisionally True, and a single
        # settle-then-wait would then block forever: the next press is
        # filtered as key-repeat, so nothing ever sets the event again.
        while True:
            self._settle()
            if self._press_evt.wait(timeout=self.RELEASE_GRACE):
                self._press_evt.clear()
                return

    def is_held(self) -> bool:
        self._settle()
        return self._held
