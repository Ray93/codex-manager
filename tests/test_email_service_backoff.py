from src.services.base import (
    BaseEmailService,
    EmailProviderBackoffState,
    EmailServiceType,
    OTPTimeoutEmailServiceError,
    RateLimitedEmailServiceError,
    apply_adaptive_backoff,
    calculate_adaptive_backoff_delay,
)


class DummyEmailService(BaseEmailService):
    def __init__(self):
        super().__init__(EmailServiceType.DUCK_MAIL, "dummy")

    def create_email(self, config=None):
        raise NotImplementedError

    def get_verification_code(
        self,
        email,
        email_id=None,
        timeout=120,
        pattern=r"(?<!\d)(\d{6})(?!\d)",
        otp_sent_at=None,
    ):
        raise NotImplementedError

    def list_emails(self, **kwargs):
        return []

    def delete_email(self, email_id: str) -> bool:
        return False

    def check_health(self) -> bool:
        return True


def test_calculate_adaptive_backoff_delay_uses_failure_count_progression():
    assert calculate_adaptive_backoff_delay(0) == 30
    assert calculate_adaptive_backoff_delay(1) == 30
    assert calculate_adaptive_backoff_delay(2) == 60
    assert calculate_adaptive_backoff_delay(3) == 120


def test_apply_adaptive_backoff_tracks_timeout_failures_to_one_hour():
    state = EmailProviderBackoffState()

    first = apply_adaptive_backoff(
        state,
        OTPTimeoutEmailServiceError("等待验证码超时", error_code="OTP_TIMEOUT_SECONDARY"),
        now=1000.0,
    )
    second = apply_adaptive_backoff(
        first,
        OTPTimeoutEmailServiceError("等待验证码超时", error_code="OTP_TIMEOUT_SECONDARY"),
        now=1031.0,
    )
    third = apply_adaptive_backoff(
        second,
        OTPTimeoutEmailServiceError("等待验证码超时", error_code="OTP_TIMEOUT_SECONDARY"),
        now=1092.0,
    )

    assert first.failures == 1
    assert first.delay_seconds == 30
    assert first.opened_until == 1030.0

    assert second.failures == 2
    assert second.delay_seconds == 60
    assert second.opened_until == 1091.0

    assert third.failures == 3
    assert third.delay_seconds == 3600
    assert third.opened_until == 4692.0


def test_apply_adaptive_backoff_keeps_normal_rate_limit_on_exponential_curve():
    state = EmailProviderBackoffState(failures=2, delay_seconds=60, opened_until=1060.0)

    next_state = apply_adaptive_backoff(
        state,
        RateLimitedEmailServiceError("请求失败: 429", retry_after=7),
        now=1100.0,
    )

    assert next_state.failures == 3
    assert next_state.delay_seconds == 120
    assert next_state.opened_until == 1220.0
    assert next_state.retry_after == 7


def test_update_status_resets_provider_backoff_after_success():
    service = DummyEmailService()

    service.update_status(False, RateLimitedEmailServiceError("请求失败: 429"))

    assert service.provider_backoff_state.failures == 1
    assert service.provider_backoff_state.delay_seconds == 30

    service.update_status(True)

    assert service.provider_backoff_state == EmailProviderBackoffState()
