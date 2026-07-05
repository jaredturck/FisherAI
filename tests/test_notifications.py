from fisher_ai.notifications import DiscordNotifier, format_embed, load_webhook_url


def test_load_webhook_url_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("STATUS_WEBHOOK", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("STATUS_WEBHOOK=https://example.test/webhook\n")

    assert load_webhook_url(env_path) == "https://example.test/webhook"


def test_notifier_posts_queued_embed_without_blocking_sender():
    notifier = DiscordNotifier("https://example.test/webhook")
    posted = []
    notifier._post = posted.append

    notifier.send(
        "Checkpoint saved",
        [("Step", "100")],
        description="Ready",
        color="green",
    )

    assert notifier.close()
    assert posted[0]["title"] == "Checkpoint saved"
    assert posted[0]["fields"][0]["name"] == "Step"


def test_embed_limits_field_count():
    fields = [(f"Field {index}", index) for index in range(30)]
    embed = format_embed("Test", fields=fields)

    assert len(embed["fields"]) == 25
