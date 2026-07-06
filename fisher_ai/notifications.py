"""Report completed training iterations to Discord."""

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBHOOK_NAME = "STATUS_WEBHOOK"


def load_webhook_url(path=None):
    """Load the Discord webhook URL from the environment file."""
    path = Path(path or PROJECT_ROOT / ".env")
    if not path.exists():
        return ""

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        if name.strip() != WEBHOOK_NAME:
            continue

        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        return value

    return ""


def collect_gpu_telemetry(device):
    """Collect a compact GPU status summary with nvidia-smi."""
    if not device.startswith("cuda"):
        return "GPU telemetry unavailable on CPU"

    index = device.split(":", 1)[1] if ":" in device else "0"
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=name,temperature.gpu,power.draw,power.limit,"
        "utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return "GPU telemetry unavailable"

    row = next(csv.reader(result.stdout.splitlines()), [])
    if len(row) != 7:
        return "GPU telemetry unavailable"

    (
        name,
        temperature,
        power_draw,
        power_limit,
        utilization,
        memory_used,
        memory_total,
    ) = [value.strip() for value in row]
    return (
        f"{name} • {utilization}% utilization • {temperature}°C\n"
        f"{power_draw} / {power_limit} W • "
        f"{memory_used} / {memory_total} MiB"
    )


class DiscordNotifier:
    """Send one Discord status report after each training iteration."""

    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url or load_webhook_url()

    def send_iteration(
        self,
        iteration,
        window_positions,
        generation,
        training,
        device,
    ):
        """Post one completed-iteration report to Discord."""
        if not self.webhook_url:
            return

        payload = {
            "username": "Fisher AI",
            "allowed_mentions": {"parse": []},
            "embeds": [
                {
                    "title": "Fisher AI iteration complete",
                    "color": 0x2ECC71,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fields": [
                        {
                            "name": "Iteration",
                            "value": f"{iteration:,}",
                            "inline": True,
                        },
                        {
                            "name": "Window positions",
                            "value": f"{window_positions:,}",
                            "inline": True,
                        },
                        {
                            "name": "Generation",
                            "value": (
                                f"{generation['elapsed_seconds']:.1f}s • "
                                f"{generation['positions_per_second']:.1f} "
                                "positions/s"
                            ),
                            "inline": False,
                        },
                        {
                            "name": "Training",
                            "value": (
                                f"{training['elapsed_seconds']:.1f}s • "
                                f"{training['positions_per_second']:.1f} "
                                "positions/s • "
                                f"{training['epochs']} epochs"
                            ),
                            "inline": False,
                        },
                        {
                            "name": "GPU",
                            "value": collect_gpu_telemetry(device),
                            "inline": False,
                        },
                    ],
                }
            ],
        }
        request = Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "FisherAI/1.0",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=5) as response:
                response.read()
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            print(f"Discord notification failed: {error}", flush=True)
