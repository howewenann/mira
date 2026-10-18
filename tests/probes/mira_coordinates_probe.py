"""Standalone Windows PyAutoGUI coordinate probe for MIRA.

Run from MIRA's Python environment:
    python mira_coordinates_probe.py

Creates its own test window and clicks only inside that window.
Screenshots are inspected in memory, never saved to disk.
Press Escape to exit. Does not use any LLM or MIRA internals.
"""

import ctypes
import platform
import tkinter as tk
from importlib.metadata import PackageNotFoundError, version

try:
    import pyautogui as pg
except ImportError as exc:
    raise SystemExit("PyAutoGUI isn't installed in this Python environment.") from exc


MAGENTA = (255, 0, 204)
pg.FAILSAFE = True  # Mouse to top-left corner to stop PyAutoGUI calls.


def windows_info():
    if platform.system() != "Windows":
        return "Windows DPI: not applicable"
    lines = []
    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()
        lines.append(f"Windows system DPI: {dpi} ({dpi / 96 * 100:.0f}% equivalent)")
    except Exception:
        lines.append("Windows system DPI: unavailable")
    try:
        value = ctypes.c_int()
        result = ctypes.windll.shcore.GetProcessDpiAwareness(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(value)
        )
        labels = {0: "unaware", 1: "system-aware", 2: "per-monitor aware"}
        lines.append(
            f"Process DPI awareness: {labels.get(value.value, value.value)}"
            if result == 0 else f"Process DPI awareness: unavailable ({result})"
        )
    except Exception:
        lines.append("Process DPI awareness: unavailable")
    return "\n".join(lines)


class CoordinateProbe:
    def __init__(self):
        self.lines = []
        self.results = []
        self.step = 0
        self.mouse_event = None
        self.running = True
        self.targets = [(0.17, 0.22), (0.50, 0.50), (0.83, 0.78)]
        self.root = tk.Tk()
        self.root.title("MIRA - Mouse Coordinate Probe (No Notepad)")
        sw, sh = pg.size()
        ww, wh = max(440, int(sw * 0.84)), max(380, int(sh * 0.80))
        ww, wh = min(ww, sw), min(wh, sh)
        x, y = max(0, (sw - ww) // 2), max(0, (sh - wh) // 2)
        self.root.geometry(f"{ww}x{wh}+{x}+{y}")
        self.root.attributes("-topmost", True)
        self.root.bind("<Escape>", lambda _e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(
            self.root,
            text="Automatic coordinate test: 3 clicks in THIS window only. Press Esc to exit.",
            font=("Segoe UI", 11, "bold"),
            pady=8,
        ).pack()
        self.status = tk.Label(self.root, text="Preparing...", font=("Segoe UI", 10))
        self.status.pack()
        self.canvas = tk.Canvas(self.root, background="#1e2530", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=(0, 8))
        self.report = tk.Text(bottom, height=9, font=("Consolas", 9), wrap="none")
        self.report.pack(fill="x", expand=False)
        controls = tk.Frame(bottom)
        controls.pack(fill="x", pady=(5, 0))
        tk.Button(controls, text="Copy report", command=self.copy_report).pack(side="left")
        tk.Button(controls, text="Close / Esc", command=self.close).pack(side="right")

        self.root.update_idletasks()
        self.log(f"Python: {platform.python_version()} | OS: {platform.platform()}")
        try:
            self.log(f"PyAutoGUI: {version('PyAutoGUI')}")
        except PackageNotFoundError:
            pass
        self.log(f"PyAutoGUI screen: {tuple(pg.size())}")
        try:
            self.shot_size = pg.screenshot().size
            self.log(f"Screenshot pixels: {self.shot_size}")
            self.log(
                "Screen/screenshot sizes: "
                + ("MATCH" if tuple(pg.size()) == self.shot_size else "MISMATCH")
            )
        except Exception as exc:
            self.shot_size = None
            self.log(f"Screenshot FAILED: {type(exc).__name__}: {exc}")
        for line in windows_info().splitlines():
            self.log(line)
        self.root.after(1600, self.run_next)

    def log(self, value):
        print(value, flush=True)
        self.lines.append(value)
        self.report.insert("end", value + "\n")
        self.report.see("end")

    def copy_report(self):
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(self.lines))
        self.root.update()
        self.status.config(text="Report copied. Paste it into ChatGPT.")

    def close(self):
        self.running = False
        self.root.destroy()

    def on_click(self, event):
        if self.running:
            self.mouse_event = (event.x_root, event.y_root)

    def draw_targets(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        r = max(22, min(65, int(w * 0.06), int(h * 0.13)))
        self.centres = []
        for i, (fx, fy) in enumerate(self.targets):
            px, py = int(w * fx), int(h * fy)
            active = i == self.step
            color = "#ff00cc" if active else "#556070"
            c.create_rectangle(px - r, py - r, px + r, py + r,
                               fill=color, outline="white", width=2)
            c.create_text(px, py - r - 14, text=f"Target {i + 1}",
                          fill="white", font=("Segoe UI", 12, "bold"))
            self.centres.append((c.winfo_rootx() + px, c.winfo_rooty() + py, r))
        c.update_idletasks()

    def run_next(self):
        if not self.running:
            return
        if self.step >= len(self.targets):
            passed = sum(self.results)
            self.log(f"RESULT: {passed}/{len(self.targets)} automated clicks landed correctly")
            if passed == len(self.targets) and self.shot_size == tuple(pg.size()):
                self.log("INTERPRETATION: Basic PyAutoGUI coordinates appear OK.")
                self.log("If MIRA misses clicks, examine model-selected coordinates,")
                self.log("screenshot resizing, and MIRA's click-tool arguments.")
            else:
                self.log("INTERPRETATION: Potential coordinate/DPI/screenshot mismatch.")
                self.log("Check the per-target readings and Windows display scaling.")
            self.status.config(text="Finished. Click 'Copy report' and paste here.")
            return

        self.draw_targets()
        self.root.lift()
        self.root.update_idletasks()
        x, y, r = self.centres[self.step]
        self.status.config(text=f"Testing target {self.step + 1}/3 at screen coordinate ({x}, {y})")
        self.mouse_event = None

        try:
            shot = pg.screenshot()  # In memory; no file is written.
            rgb = tuple(shot.convert("RGB").getpixel((x, y)))
            pixel_ok = rgb == MAGENTA
            self.log(f"Target {self.step + 1}: requested=({x},{y}), screenshot RGB={rgb}, "
                     f"expected magenta={pixel_ok}")
        except Exception as exc:
            self.log(f"Target {self.step + 1}: screenshot sample FAILED: "
                     f"{type(exc).__name__}: {exc}")
            pixel_ok = False

        self.pixel_ok = pixel_ok
        try:
            pg.click(x=x, y=y)
            self.log(f"    after click: PyAutoGUI mouse={tuple(pg.position())}")
        except Exception as exc:
            self.log(f"    click FAILED: {type(exc).__name__}: {exc}")
        self.root.after(350, self.finish_step)

    def finish_step(self):
        if not self.running:
            return
        x, y, r = self.centres[self.step]
        event = self.mouse_event
        # The click must be received by the Tk canvas, near the intended target.
        hit = event is not None and abs(event[0] - x) <= r and abs(event[1] - y) <= r
        good = bool(hit and self.pixel_ok)
        self.results.append(good)
        self.log(f"    Tk received click at: {event}; target hit: {bool(hit)}; "
                 f"overall: {'PASS' if good else 'FAIL'}")
        self.step += 1
        self.root.after(700, self.run_next)

    def start(self):
        self.root.mainloop()


if __name__ == "__main__":
    CoordinateProbe().start()
