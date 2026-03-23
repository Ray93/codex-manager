import src.services.tempmail as tempmail_module
from src.services.tempmail import TempmailService


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeHTTPClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"未准备响应: GET {url}")
        return self.responses.pop(0)


def test_get_verification_code_ignores_messages_not_newer_than_otp_anchor(monkeypatch):
    service = TempmailService({
        "base_url": "https://api.tempmail.test/v2",
        "timeout": 1,
        "max_retries": 1,
    })
    service._email_cache["tester@example.com"] = {
        "email": "tester@example.com",
        "token": "token-1",
    }
    service.http_client = FakeHTTPClient([
        FakeResponse(
            status_code=200,
            payload={
                "emails": [
                    {
                        "id": "old-mail",
                        "from": "noreply@openai.com",
                        "subject": "Old verification code",
                        "body": "111111",
                        "date": 1999,
                    },
                    {
                        "id": "new-mail",
                        "from": "noreply@openai.com",
                        "subject": "New verification code",
                        "body": "654321",
                        "date": 2001,
                    },
                ]
            },
        )
    ])
    monkeypatch.setattr(tempmail_module.time, "sleep", lambda _: None)

    code = service.get_verification_code(
        email="tester@example.com",
        timeout=1,
        otp_sent_at=2000,
    )

    assert code == "654321"
    assert service.http_client.calls == [
        {
            "url": "https://api.tempmail.test/v2/inbox",
            "kwargs": {
                "params": {"token": "token-1"},
                "headers": {"Accept": "application/json"},
            },
        }
    ]
