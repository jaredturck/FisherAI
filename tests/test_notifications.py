import json

from fisher_ai import notifications
from fisher_ai.notifications import DiscordNotifier, load_webhook_url


def test_load_webhook_url_from_env_file(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("STATUS_WEBHOOK=https://example.test/webhook\n")

    assert load_webhook_url(env_path) == "https://example.test/webhook"


def test_notifier_posts_one_iteration_payload(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b""

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(notifications, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        notifications,
        "collect_gpu_telemetry",
        lambda device: "GPU ready",
    )

    notifier = DiscordNotifier("https://example.test/webhook")
    notifier.send_iteration(
        2,
        50000,
        {"elapsed_seconds": 10.0, "positions_per_second": 5000.0},
        {
            "elapsed_seconds": 2.0,
            "positions_per_second": 75000.0,
            "epochs": 3,
        },
        "cuda:0",
    )

    payload = json.loads(requests[0][0].data)
    fields = payload["embeds"][0]["fields"]
    assert requests[0][1] == 5
    assert fields[0]["value"] == "2"
    assert fields[-1]["value"] == "GPU ready"
