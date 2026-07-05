import csv
import json
import os
import queue
import socket
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBED_COLORS = {
    "blue": 0x3498DB,
    "green": 0x2ECC71,
    "orange": 0xF39C12,
    "red": 0xE74C3C,
}

GPU_QUERY_ARGUMENTS = [
    "--query-gpu=index,name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def load_env_value(path, name):
    path = Path(path)
    if not path.exists():
        return ""

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() != name:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value

    return ""


def load_webhook_url(env_path=None):
    webhook_url = os.environ.get("STATUS_WEBHOOK", "").strip()
    if webhook_url:
        return webhook_url

    path = env_path or PROJECT_ROOT / ".env"
    return load_env_value(path, "STATUS_WEBHOOK").strip()


def run_nvidia_smi(arguments, timeout=3):
    result = subprocess.run(
        ["nvidia-smi", *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def parse_gpu_query(output):
    gpus = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 8:
            continue

        values = [value.strip() for value in row]
        gpus.append(
            {
                "index": values[0],
                "name": values[1],
                "temperature": values[2],
                "power_draw": values[3],
                "power_limit": values[4],
                "utilization": values[5],
                "memory_used": values[6],
                "memory_total": values[7],
            }
        )

    return gpus


def collect_gpu_telemetry():
    try:
        return parse_gpu_query(run_nvidia_smi(GPU_QUERY_ARGUMENTS)), None
    except (
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        return [], str(error)


def format_metric(value, suffix):
    if value in {"", "N/A", "[Not Supported]"}:
        return "Unavailable"

    try:
        number = float(value)
    except ValueError:
        return f"{value}{suffix}"

    if number.is_integer():
        value = f"{int(number):,}"
    else:
        value = f"{number:,.1f}"
    return f"{value}{suffix}"


def gpu_telemetry_fields(gpus, error=None):
    if not gpus:
        return [("GPU telemetry", clean_text(error or "Unavailable", 1024), False)]

    fields = []
    for gpu in gpus:
        value = (
            f"Utilization: **{format_metric(gpu['utilization'], '%')}** • "
            f"Temperature: **{format_metric(gpu['temperature'], '°C')}**\n"
            f"Power: **{format_metric(gpu['power_draw'], ' W')} / "
            f"{format_metric(gpu['power_limit'], ' W')}** • "
            f"Memory: **{format_metric(gpu['memory_used'], ' MiB')} / "
            f"{format_metric(gpu['memory_total'], ' MiB')}**"
        )
        fields.append((f"GPU {gpu['index']} — {gpu['name']}", value, False))

    return fields


def clean_text(value, maximum_length):
    text = str(value)
    if len(text) <= maximum_length:
        return text
    return text[: maximum_length - 1] + "…"


def format_embed(title, fields=None, description=None, color="blue"):
    embed = {
        "title": clean_text(title, 256),
        "color": EMBED_COLORS.get(color, EMBED_COLORS["blue"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {
            "text": clean_text(
                f"Fisher AI Training Monitor • {socket.gethostname()}",
                2048,
            )
        },
    }

    if description:
        embed["description"] = clean_text(description, 4096)

    if fields:
        embed["fields"] = []
        for field in fields[:25]:
            if len(field) == 2:
                name, value = field
                inline = True
            else:
                name, value, inline = field

            embed["fields"].append(
                {
                    "name": clean_text(name, 256),
                    "value": clean_text(value, 1024),
                    "inline": bool(inline),
                }
            )

    return embed


class DiscordNotifier:
    def __init__(self, webhook_url=None, request_timeout_seconds=5):
        self.webhook_url = webhook_url or load_webhook_url()
        self.request_timeout_seconds = request_timeout_seconds
        self.enabled = bool(self.webhook_url)
        self.messages = queue.Queue()
        self.worker = None
        self.last_error = None

        if self.enabled:
            self.worker = threading.Thread(
                target=self._run,
                name="fisher-discord-notifier",
                daemon=True,
            )
            self.worker.start()

    def send(
        self,
        title,
        fields=None,
        description=None,
        color="blue",
        include_gpu_stats=False,
    ):
        if not self.enabled:
            return

        self.messages.put(
            {
                "title": title,
                "fields": fields,
                "description": description,
                "color": color,
                "include_gpu_stats": include_gpu_stats,
            }
        )

    def _run(self):
        while True:
            event = self.messages.get()
            if event is None:
                self.messages.task_done()
                return

            include_gpu_stats = event.pop("include_gpu_stats")
            if include_gpu_stats:
                gpus, error = collect_gpu_telemetry()
                event["fields"] = list(event.get("fields") or []) + gpu_telemetry_fields(
                    gpus,
                    error,
                )

            self._post(format_embed(**event))
            self.messages.task_done()

    def _post(self, embed):
        payload = {
            "username": "Fisher AI",
            "embeds": [embed],
            "allowed_mentions": {"parse": []},
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
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            self.last_error = str(error)
            print(f"Discord notification failed: {error}", flush=True)

    def close(self):
        if not self.enabled or self.worker is None:
            return self.last_error is None

        self.messages.put(None)
        self.worker.join(timeout=self.request_timeout_seconds + 1)
        return self.last_error is None and not self.worker.is_alive()
