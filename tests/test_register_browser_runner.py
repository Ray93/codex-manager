from src.config.constants import EmailServiceType
from src.core.register_browser import BrowserRegistrationRunner
from src.services.base import BaseEmailService


class DummyEmailService(BaseEmailService):
    def __init__(self):
        super().__init__(EmailServiceType.TEMPMAIL, "dummy")

    def create_email(self, config=None):
        return {"email": "tester@example.com", "service_id": "svc-1"}

    def get_verification_code(
        self,
        email,
        email_id=None,
        timeout=60,
        pattern=r"(?<!\d)(\d{6})(?!\d)",
        otp_sent_at=None,
    ):
        return "123456"

    def list_emails(self, **kwargs):
        return []

    def delete_email(self, email_id):
        return True

    def check_health(self):
        return True


class FakeLocator:
    def __init__(self, page, selector):
        self.page = page
        self.selector = selector

    def count(self):
        return 1 if self.selector in self.page.visible else 0

    @property
    def first(self):
        return self

    def is_visible(self):
        return self.selector in self.page.visible

    def click(self):
        self.page.clicked.append(self.selector)
        if self.selector == "a[href*='login_with']":
            self.page.visible.discard(self.selector)
            self.page.visible.add("input[type='email']")


class FakePage:
    def __init__(self):
        self.visible = {"a[href*='login_with']"}
        self.clicked = []
        self.waits = []
        self.url = "https://auth.openai.com/session-ended"
        self.stage_after_waits = {}

    def locator(self, selector):
        return FakeLocator(self, selector)

    def wait_for_timeout(self, ms):
        self.waits.append(ms)
        next_visible = self.stage_after_waits.get(len(self.waits))
        if next_visible is not None:
            self.visible = set(next_visible)


class ProfileLocator:
    def __init__(self, page, selector):
        self.page = page
        self.selector = selector

    def count(self):
        return 1 if self.selector in self.page.available else 0

    @property
    def first(self):
        return self

    def is_visible(self):
        return self.selector in self.page.visible

    def fill(self, value):
        self.page.filled.append((self.selector, value))

    def select_option(self, value=None):
        self.page.selected.append((self.selector, value))

    def evaluate(self, script, value):
        self.page.evaluated.append((self.selector, value))


class ProfilePage:
    def __init__(self):
        self.available = {"input[name='name']", "input[name='birthday']"}
        self.visible = {"input[name='name']"}
        self.filled = []
        self.selected = []
        self.evaluated = []

    def locator(self, selector):
        return ProfileLocator(self, selector)


def _build_runner():
    return BrowserRegistrationRunner(
        auth_url="https://auth.openai.com/oauth/authorize",
        redirect_uri="http://localhost:1455/auth/callback",
        email="tester@example.com",
        email_service=DummyEmailService(),
        email_info={"service_id": "svc-1"},
        password="Password123!",
        user_info={"name": "Tester", "birthdate": "1995-01-01"},
        headless=True,
        timeout_seconds=5,
    )


def test_wait_for_login_page_enters_via_session_ended_link():
    runner = _build_runner()
    page = FakePage()

    runner._wait_for_login_page(page, timeout_ms=1_000)

    assert page.clicked == ["a[href*='login_with']"]
    assert runner._is_visible(page, runner._email_selectors()) is True


def test_maybe_switch_to_signup_page_clicks_register_link_from_login():
    runner = _build_runner()
    page = FakePage()
    page.url = "https://auth.openai.com/log-in"
    page.visible = {"input[type='email']", "a[href='/create-account']"}

    runner._maybe_switch_to_signup_page(page)

    assert page.clicked == ["a[href='/create-account']"]


def test_should_retry_headed_on_cloudflare_verification_page():
    runner = _build_runner()

    assert (
        runner._should_retry_headed(
            "https://auth.openai.com/api/oauth/oauth2/auth?foo=bar",
            "auth.openai.com\n执行安全验证\n此网站使用安全服务来防范恶意自动程序",
        )
        is True
    )
    assert (
        runner._should_retry_headed(
            "https://auth.openai.com/log-in",
            "欢迎回来\n电子邮件地址\n继续",
        )
        is False
    )


def test_wait_for_post_email_stage_waits_until_password_appears():
    runner = _build_runner()
    page = FakePage()
    page.url = "https://auth.openai.com/create-account"
    page.visible = set()
    page.stage_after_waits = {
        2: {"input[type='password']"},
    }

    stage = runner._wait_for_post_email_stage(page, timeout_ms=1_000)

    assert stage == "password"


def test_complete_profile_step_sets_hidden_birthday_input_when_present():
    runner = _build_runner()
    page = ProfilePage()
    runner.user_info = {"name": "Tester", "birthdate": "1995-05-16"}

    runner._complete_profile_step(page)

    assert ("input[name='name']", "Tester") in page.filled
    assert ("input[name='birthday']", "1995-05-16") in page.evaluated


def test_complete_profile_step_fills_birthday_spinbutton_segments_when_present():
    runner = _build_runner()
    page = ProfilePage()
    page.available = {
        "input[name='name']",
        "[role='spinbutton'][data-type='year']",
        "[role='spinbutton'][data-type='month']",
        "[role='spinbutton'][data-type='day']",
    }
    page.visible = set(page.available)
    runner.user_info = {"name": "Tester", "birthdate": "1995-05-16"}

    runner._complete_profile_step(page)

    assert ("[role='spinbutton'][data-type='year']", "1995") in page.filled
    assert ("[role='spinbutton'][data-type='month']", "05") in page.filled
    assert ("[role='spinbutton'][data-type='day']", "16") in page.filled


def test_handle_post_profile_stage_restarts_login_when_add_phone_detected(monkeypatch):
    runner = _build_runner()
    page = FakePage()
    page.url = "https://auth.openai.com/add-phone"
    captured = {"url": ""}
    calls = []

    monkeypatch.setattr(runner, "_wait_for_callback", lambda *args, **kwargs: False)
    monkeypatch.setattr(runner, "_restart_login_flow", lambda *args, **kwargs: calls.append("restart"))

    runner._handle_post_profile_stage(page, captured)

    assert calls == ["restart"]
