"""Scope Linux kernel fault evidence to the serving devices and trial interval."""

import os
import re
import subprocess


def device_fault(line, devices):
    """Match a fault to a complete PCI address on the same kernel log line."""
    addresses = re.findall(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b", line.lower())
    fault = re.search(
        r"Engine reset|Timedout job|GPU HANG|devcoredump|Engine memory CAT error|NVRM: Xid", line, re.I
    )
    # NVIDIA Xid lines can omit the PCI function.
    if "NVRM: Xid" in line:
        addresses.extend(bdf + ".0" for bdf in re.findall(r"PCI:([0-9a-f:]+)\)", line.lower(), re.I))
    return bool(set(addresses) & set(devices) and fault)


def journal(*args):
    try:
        proc = subprocess.run(
            ["journalctl", "-k", "--no-pager", *args], capture_output=True, text=True, timeout=10, check=True
        )
        return proc.stdout
    except (OSError, subprocess.SubprocessError):
        return None


class DeviceJournal:
    def __init__(self, devices, logfile=None):
        self.devices = devices
        self.logfile = logfile
        try:
            self.offset = logfile.stat() if logfile else None
        except OSError:
            self.offset = None
        text = journal("-n", "1", "--show-cursor") if devices and not logfile else None
        match = re.search(r"^-- cursor: (.+)$", text or "", re.M)
        self.cursor = match[1] if match else None

    def finish(self, path):
        if not self.devices:
            return None
        if self.logfile:
            if not self.offset:
                return None
            try:
                log = self.logfile.open(errors="replace")
            except OSError:
                return None
            with log as f:
                current = os.fstat(f.fileno())
                if current.st_ino != self.offset.st_ino or current.st_size < self.offset.st_size:
                    return None
                f.seek(self.offset.st_size)
                text = f.read()
        elif self.cursor:
            text = journal("--after-cursor", self.cursor)
        else:
            return None
        if text is None:
            return None
        path.write_text(text)
        return not any(device_fault(line, self.devices) for line in text.splitlines())
