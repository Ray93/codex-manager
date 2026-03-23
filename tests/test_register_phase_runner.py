from types import SimpleNamespace

import src.core.register as register_module
from src.core.register import PhaseContext, PhaseResult, RegistrationEngine, RegistrationResult
from src.services import EmailServiceType
from src.services.base import BaseEmailService, RateLimitedEmailServiceError


class DummySettings:
    openai_client_id = "client-id"
    openai_auth_url = "https://auth.example.test"
    openai_token_url = "https://token.example.test"
    openai_redirect_uri = "https://callback.example.test"
    openai_scope = "openid profile email"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeCookies(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class FakePasswordSession:
    def __init__(self, response):
        self.response = response
        self.cookies = FakeCookies({"oai-did": "did-1"})
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        return self.response

    def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        return FakeResponse(status_code=200)


def _build_engine(monkeypatch):
    monkeypatch.setattr(register_module, "get_settings", lambda: DummySettings())
    email_service = SimpleNamespace(service_type=EmailServiceType.DUCK_MAIL)
    return RegistrationEngine(email_service=email_service)


class RateLimitedEmailService(BaseEmailService):
    def __init__(self):
        super().__init__(EmailServiceType.DUCK_MAIL, "duck-test")

    def create_email(self, config=None):
        error = RateLimitedEmailServiceError("请求失败: 429", retry_after=7)
        self.update_status(False, error)
        raise error

    def get_verification_code(self, email, email_id=None, timeout=120, pattern=r"(?<!\d)(\d{6})(?!\d)", otp_sent_at=None):
        raise NotImplementedError

    def list_emails(self, **kwargs):
        return []

    def delete_email(self, email_id: str) -> bool:
        return False

    def check_health(self) -> bool:
        return True


def test_run_executes_nine_explicit_phases(monkeypatch):
    engine = _build_engine(monkeypatch)
    order = []
    phase_names = [
        "ip_check",
        "email_prepare",
        "signup",
        "otp_primary",
        "account_create",
        "oauth_reenter",
        "otp_secondary",
        "workspace_resolve",
        "oauth_callback",
    ]

    def make_phase(name):
        def _phase(result, context):
            order.append(name)
            if name == "email_prepare":
                result.email = "tester@example.com"
            if name == "workspace_resolve":
                result.workspace_id = "ws-1"
                context.callback_url = "https://callback.example.test?code=abc&state=xyz"
            if name == "oauth_callback":
                context.token_info = {
                    "account_id": "acct-1",
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "id_token": "id-token",
                }
            return PhaseResult(phase=name, success=True, data={"phase": name})

        return _phase

    for phase_name in phase_names:
        monkeypatch.setattr(engine, f"_phase_{phase_name}", make_phase(phase_name))

    engine.session = SimpleNamespace(
        cookies=FakeCookies({"__Secure-next-auth.session-token": "session-token"})
    )

    result = engine.run()

    assert result.success is True
    assert order == phase_names
    assert [item.phase for item in engine.phase_history] == phase_names
    assert all(isinstance(item, PhaseResult) for item in engine.phase_history)
    assert result.email == "tester@example.com"
    assert result.workspace_id == "ws-1"
    assert result.account_id == "acct-1"
    assert result.session_token == "session-token"
    assert result.source == "register"
    assert "registration_mode" not in result.metadata


def test_run_stops_on_first_failed_phase(monkeypatch):
    engine = _build_engine(monkeypatch)
    order = []

    def success_phase(name):
        def _phase(result, context):
            order.append(name)
            return PhaseResult(phase=name, success=True)

        return _phase

    def failed_signup(result, context):
        order.append("signup")
        return PhaseResult(
            phase="signup",
            success=False,
            error_message="提交注册表单失败: 协议错误",
        )

    monkeypatch.setattr(engine, "_phase_ip_check", success_phase("ip_check"))
    monkeypatch.setattr(engine, "_phase_email_prepare", success_phase("email_prepare"))
    monkeypatch.setattr(engine, "_phase_signup", failed_signup)

    for phase_name in ["otp_primary", "account_create", "oauth_reenter", "otp_secondary", "workspace_resolve", "oauth_callback"]:
        monkeypatch.setattr(
            engine,
            f"_phase_{phase_name}",
            lambda result, context, name=phase_name: PhaseResult(
                phase=name,
                success=True,
            ),
        )

    result = engine.run()

    assert result.success is False
    assert result.error_message == "提交注册表单失败: 协议错误"
    assert order == ["ip_check", "email_prepare", "signup"]
    assert [item.phase for item in engine.phase_history] == order


def test_email_prepare_phase_exposes_provider_backoff(monkeypatch):
    monkeypatch.setattr(register_module, "get_settings", lambda: DummySettings())
    engine = RegistrationEngine(email_service=RateLimitedEmailService())

    phase_result = engine._phase_email_prepare(
        RegistrationResult(success=False, logs=[]),
        PhaseContext(),
    )

    assert phase_result.success is False
    assert phase_result.error_code == "EMAIL_PROVIDER_RATE_LIMITED"
    assert phase_result.retryable is True
    assert phase_result.next_action == "switch_provider"
    assert phase_result.provider_backoff is not None
    assert phase_result.provider_backoff.failures == 1
    assert phase_result.provider_backoff.delay_seconds == 30
    assert phase_result.provider_backoff.retry_after == 7


def test_submit_login_password_step_returns_continue_url(monkeypatch):
    engine = _build_engine(monkeypatch)
    engine.email = "tester@example.com"
    engine.password = "Pass12345"
    engine.session = FakePasswordSession(
        FakeResponse(status_code=200, payload={"continue_url": "https://continue.example.test"})
    )
    monkeypatch.setattr(engine, "_check_sentinel", lambda did: None)

    step_result = engine._submit_login_password_step()

    assert step_result.success is True
    assert step_result.http_status == 200
    assert step_result.continue_url == "https://continue.example.test"
    assert engine.session.get_calls == [
        {"url": "https://continue.example.test", "timeout": 15}
    ]


def test_otp_secondary_timeout_uses_independent_anchor_and_returns_explicit_error(monkeypatch):
    engine = _build_engine(monkeypatch)
    engine._is_existing_account = False
    engine._otp_sent_at = 100.0

    captured = {}

    monkeypatch.setattr(
        register_module.time,
        "time",
        lambda: 500.0,
    )
    monkeypatch.setattr(
        engine,
        "_submit_login_password_step",
        lambda: SimpleNamespace(success=True, http_status=200),
    )

    def fake_get_verification_code(otp_sent_at=None, timeout=120):
        captured["otp_sent_at"] = otp_sent_at
        captured["timeout"] = timeout
        return None

    monkeypatch.setattr(engine, "_get_verification_code", fake_get_verification_code)

    phase_result = engine._phase_otp_secondary(
        RegistrationResult(success=False, logs=[]),
        PhaseContext(reenter_ready=True),
    )

    assert captured == {
        "otp_sent_at": 500.0,
        "timeout": 120,
    }
    assert phase_result.success is False
    assert phase_result.error_code == "OTP_TIMEOUT_SECONDARY"
    assert phase_result.error_message == "等待第二次验证码超时"
    assert phase_result.retryable is True
    assert phase_result.next_action == "extend_timeout"
    assert phase_result.data == {"otp_sent_at": 500.0}
