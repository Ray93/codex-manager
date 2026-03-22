import base64
import json

from src.config.constants import EmailServiceType
from src.core.register import RegistrationEngine
from src.services.base import BaseEmailService


def _encode_cookie_segment(payload):
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class DummyEmailService(BaseEmailService):
    def __init__(self):
        super().__init__(EmailServiceType.TEMPMAIL, "dummy")

    def create_email(self, config=None):
        raise NotImplementedError

    def get_verification_code(
        self,
        email,
        email_id=None,
        timeout=60,
        pattern=r"(?<!\d)(\d{6})(?!\d)",
        otp_sent_at=None,
    ):
        raise NotImplementedError

    def list_emails(self, **kwargs):
        return []

    def delete_email(self, email_id):
        raise NotImplementedError

    def check_health(self):
        return True


class DummyCookies:
    def __init__(self, values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


class DummySession:
    def __init__(self, values):
        self.cookies = DummyCookies(values)


def _build_engine(cookie_values):
    engine = RegistrationEngine(email_service=DummyEmailService())
    engine.session = DummySession(cookie_values)
    return engine


def test_get_workspace_id_reads_payload_segment_from_auth_session_cookie():
    auth_cookie = ".".join(
        [
            _encode_cookie_segment({"alg": "HS256", "typ": "JWT"}),
            _encode_cookie_segment({"workspaces": [{"id": "ws_from_payload"}]}),
            "signature",
        ]
    )
    engine = _build_engine({"oai-client-auth-session": auth_cookie})

    workspace_id = engine._get_workspace_id()

    assert workspace_id == "ws_from_payload"


def test_get_workspace_id_falls_back_to_auth_info_cookie():
    auth_info_cookie = _encode_cookie_segment(
        {"workspaces": [{"id": "ws_from_auth_info"}]}
    )
    engine = _build_engine({"oai_client_auth_info": auth_info_cookie})

    workspace_id = engine._get_workspace_id()

    assert workspace_id == "ws_from_auth_info"
