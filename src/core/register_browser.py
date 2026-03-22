"""
Browser-based OpenAI registration flow driven by Playwright.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from ..services.base import BaseEmailService

logger = logging.getLogger(__name__)


@dataclass
class BrowserRegistrationArtifacts:
    callback_url: str
    session_token: str = ""
    cookies: str = ""
    workspace_id: str = ""
    is_existing_account: bool = True
    password_used: str = ""


class BrowserRegistrationRunner:
    def __init__(
        self,
        *,
        auth_url: str,
        redirect_uri: str,
        email: str,
        email_service: BaseEmailService,
        email_info: Optional[Dict[str, Any]],
        password: str,
        user_info: Dict[str, Any],
        headless: bool = True,
        timeout_seconds: int = 120,
        proxy_url: Optional[str] = None,
        logger_callback: Optional[Callable[[str], None]] = None,
    ):
        self.auth_url = auth_url
        self.redirect_uri = redirect_uri
        self.email = email
        self.email_service = email_service
        self.email_info = email_info or {}
        self.password = password
        self.user_info = user_info
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.proxy_url = proxy_url
        self.logger_callback = logger_callback or (lambda message: logger.info(message))

    def run(self) -> BrowserRegistrationArtifacts:
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is not installed in the current environment") from exc

        with sync_playwright() as playwright:
            try:
                return self._run_once(playwright, headless=self.headless)
            except RuntimeError as exc:
                if self.headless and "security verification page" in str(exc).lower():
                    self._log("浏览器流程: headless 命中安全验证页，回退到有界面 Chrome 重试")
                    return self._run_once(playwright, headless=False)
                raise

    def _run_once(self, playwright, *, headless: bool) -> BrowserRegistrationArtifacts:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        captured_callback = {"url": ""}
        launch_options = {
            "headless": headless,
            "channel": "chrome",
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        proxy = self._playwright_proxy()
        if proxy:
            launch_options["proxy"] = proxy

        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context(ignore_https_errors=True)
        context.set_default_timeout(self.timeout_seconds * 1000)
        context.on("request", lambda request: self._capture_callback(request.url, captured_callback))
        page = context.new_page()

        try:
            self._log("浏览器流程: 打开 OpenAI OAuth 页面")
            page.goto(self.auth_url, wait_until="domcontentloaded")
            self._wait_for_login_page(page, timeout_ms=20_000)
            self._maybe_switch_to_signup_page(page)
            self._fill_email_step(page)
            self._click_primary(page)

            artifacts = BrowserRegistrationArtifacts(callback_url="")
            next_stage = self._wait_for_post_email_stage(page, timeout_ms=15_000)

            if next_stage == "callback":
                artifacts.callback_url = captured_callback["url"] or page.url
                return self._finalize_artifacts(context, artifacts)

            if next_stage == "password":
                self._log("浏览器流程: 检测到密码步骤")
                self._fill_first(page, self._password_selectors(), self.password)
                artifacts.password_used = self.password
                self._click_primary(page)
                page.wait_for_timeout(2000)

            if next_stage in {"password", "otp"}:
                self._complete_otp_step(page)

            if self._wait_for_callback(page, captured_callback, timeout_ms=6_000):
                artifacts.callback_url = captured_callback["url"]
                return self._finalize_artifacts(context, artifacts)

            if next_stage == "profile" or self._is_visible(page, self._profile_selectors()):
                artifacts.is_existing_account = False
                self._complete_profile_step(page)
                self._click_primary(page)

            self._handle_post_profile_stage(page, captured_callback)
            artifacts.callback_url = captured_callback["url"]
            return self._finalize_artifacts(context, artifacts)
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(f"Playwright registration timeout: {exc}") from exc
        finally:
            browser.close()

    def _playwright_proxy(self) -> Optional[Dict[str, str]]:
        if not self.proxy_url:
            return None

        parsed = urlparse(self.proxy_url)
        if not parsed.scheme or not parsed.hostname or not parsed.port:
            return None

        proxy = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
        if parsed.username:
            proxy["username"] = parsed.username
        if parsed.password:
            proxy["password"] = parsed.password
        return proxy

    def _capture_callback(self, url: str, store: Dict[str, str]) -> None:
        if self._is_callback_url(url):
            store["url"] = url

    def _is_callback_url(self, url: str) -> bool:
        return bool(url and url.startswith(self.redirect_uri) and "code=" in url and "state=" in url)

    def _wait_for_callback(self, page, store: Dict[str, str], timeout_ms: int) -> bool:
        deadline = time.time() + (timeout_ms / 1000)
        while time.time() < deadline:
            current_url = page.url or ""
            if self._is_callback_url(current_url):
                store["url"] = current_url
                return True
            if store.get("url"):
                return True
            page.wait_for_timeout(250)
        return bool(store.get("url"))

    def _handle_post_profile_stage(self, page, store: Dict[str, str]) -> None:
        if self._wait_for_callback(page, store, timeout_ms=8_000):
            return

        if self._is_add_phone_page(page):
            self._log("浏览器流程: 命中 add-phone，切换到重新登录流程")
            self._restart_login_flow(page, store)
            return

        if self._is_consent_page(page):
            self._log("浏览器流程: 检测到 consent 页面，继续授权")
            self._click_primary(page)
            if self._wait_for_callback(page, store, timeout_ms=15_000):
                return

        self._wait_for_callback(page, store, timeout_ms=self.timeout_seconds * 1000)

    def _restart_login_flow(self, page, store: Dict[str, str]) -> None:
        page.goto(self.auth_url, wait_until="domcontentloaded")
        self._wait_for_login_page(page, timeout_ms=20_000)
        self._fill_email_step(page)
        self._click_primary(page)

        next_stage = self._wait_for_post_email_stage(page, timeout_ms=15_000)
        if next_stage == "callback":
            store["url"] = store.get("url") or page.url
            return

        if next_stage == "password":
            self._log("浏览器流程: 重新登录时检测到密码步骤")
            self._fill_first(page, self._password_selectors(), self.password)
            self._click_primary(page)
            page.wait_for_timeout(2_000)

        if next_stage in {"password", "otp"}:
            self._complete_otp_step(page)

        if self._wait_for_callback(page, store, timeout_ms=8_000):
            return

        if self._is_consent_page(page):
            self._log("浏览器流程: 重新登录后进入 consent 页面，继续授权")
            self._click_primary(page)
            if self._wait_for_callback(page, store, timeout_ms=15_000):
                return

        self._wait_for_callback(page, store, timeout_ms=self.timeout_seconds * 1000)

    def _session_ended_selectors(self):
        return [
            "a[href*='login_with']",
            "a:has-text('登录')",
            "button:has-text('登录')",
        ]

    def _signup_selectors(self):
        return [
            "a[href='/create-account']",
            "a[href*='create-account']",
            "a:has-text('注册')",
            "button:has-text('注册')",
        ]

    def _wait_for_login_page(self, page, timeout_ms: int) -> None:
        deadline = time.time() + (timeout_ms / 1000)
        clicked_session_link = False

        while time.time() < deadline:
            if self._is_visible(page, self._email_selectors()):
                return

            if self._is_visible(page, self._session_ended_selectors()):
                if not clicked_session_link:
                    self._log("浏览器流程: 检测到会话已结束页，跳转到登录表单")
                    self._click_first_visible(page, self._session_ended_selectors())
                    clicked_session_link = True
                page.wait_for_timeout(2_000)
                continue

            page.wait_for_timeout(250)

        current_url = getattr(page, "url", "")
        body_text = self._safe_body_text(page)
        if self._should_retry_headed(current_url, body_text):
            raise RuntimeError(
                "Hit Cloudflare security verification page while waiting for login email page "
                f"(url={current_url})"
            )
        raise RuntimeError(
            "Could not reach the login email page after opening OAuth URL "
            f"(url={current_url})"
        )

    def _wait_for_post_email_stage(self, page, timeout_ms: int) -> str:
        deadline = time.time() + (timeout_ms / 1000)

        while time.time() < deadline:
            if self._is_callback_url(getattr(page, "url", "") or ""):
                return "callback"
            if self._is_visible(page, self._password_selectors()):
                return "password"
            if self._is_visible(page, self._profile_selectors()):
                return "profile"
            if self._is_visible(page, self._otp_selectors()):
                return "otp"
            page.wait_for_timeout(250)

        current_url = getattr(page, "url", "")
        raise RuntimeError(
            "Could not determine post-email registration stage "
            f"(url={current_url})"
        )

    def _maybe_switch_to_signup_page(self, page) -> None:
        current_url = str(getattr(page, "url", "") or "")
        if "/log-in" not in current_url:
            return
        if not self._is_visible(page, self._signup_selectors()):
            return

        self._log("浏览器流程: 从登录页切换到注册页")
        if self._click_first_visible(page, self._signup_selectors()):
            page.wait_for_timeout(2_000)

    def _safe_body_text(self, page) -> str:
        try:
            return page.locator("body").inner_text(timeout=5_000)
        except Exception:
            return ""

    def _is_add_phone_page(self, page) -> bool:
        current_url = str(getattr(page, "url", "") or "")
        return "/add-phone" in current_url

    def _is_consent_page(self, page) -> bool:
        current_url = str(getattr(page, "url", "") or "")
        return "/sign-in-with-chatgpt/" in current_url

    def _should_retry_headed(self, current_url: str, body_text: str) -> bool:
        url = str(current_url or "").lower()
        text = str(body_text or "").lower()
        return (
            "api/oauth/oauth2/auth" in url
            and (
                "执行安全验证" in body_text
                or "please wait" in text
                or "security verification" in text
                or "cloudflare" in text
            )
        )

    def _fill_email_step(self, page) -> None:
        self._log(f"浏览器流程: 输入邮箱 {self.email}")
        self._fill_first(page, self._email_selectors(), self.email)

    def _complete_otp_step(self, page) -> None:
        self._log("浏览器流程: 获取并填写邮箱验证码")
        code = self.email_service.get_verification_code(
            email=self.email,
            email_id=self.email_info.get("service_id"),
            timeout=self.timeout_seconds,
        )
        if not code:
            raise RuntimeError("Email verification code was not received")

        if self._is_visible(page, ["input[autocomplete='one-time-code']", "input[inputmode='numeric']"]):
            otp_inputs = page.locator("input[autocomplete='one-time-code'], input[inputmode='numeric']")
            count = otp_inputs.count()
            if count >= len(code):
                for index, digit in enumerate(code):
                    otp_inputs.nth(index).fill(digit)
            else:
                otp_inputs.first.fill(code)
        else:
            self._fill_first(page, ["input[type='tel']", "input[type='text']", "input[type='number']"], code)

        self._click_primary(page)

    def _complete_profile_step(self, page) -> None:
        self._log("浏览器流程: 填写 about-you 信息")
        name = str(self.user_info.get("name") or "").strip()
        birthdate = str(self.user_info.get("birthdate") or "").strip()
        self._fill_first(page, self._profile_selectors(), name)

        if birthdate:
            year, month, day = birthdate.split("-")

            if self._fill_if_visible(page, self._birthday_segment_selectors("year"), year):
                self._fill_if_visible(page, self._birthday_segment_selectors("month"), month)
                self._fill_if_visible(page, self._birthday_segment_selectors("day"), day)
                return

            if self._set_input_value(page, ["input[name='birthday']"], birthdate):
                return

        if self._is_visible(page, ["input[type='date']"]):
            self._fill_first(page, ["input[type='date']"], birthdate)
            return

        if birthdate:
            self._select_if_visible(page, ["select[name='month']", "select[aria-label*='Month']"], str(int(month)))
            self._select_if_visible(page, ["select[name='day']", "select[aria-label*='Day']"], str(int(day)))
            self._select_if_visible(page, ["select[name='year']", "select[aria-label*='Year']"], year)

    def _finalize_artifacts(self, context, artifacts: BrowserRegistrationArtifacts) -> BrowserRegistrationArtifacts:
        cookies = context.cookies()
        artifacts.cookies = self._serialize_cookies(cookies)
        artifacts.session_token = self._extract_cookie(cookies, "__Secure-next-auth.session-token")
        return artifacts

    def _serialize_cookies(self, cookies) -> str:
        parts = []
        for cookie in cookies:
            name = str(cookie.get("name") or "").strip()
            value = str(cookie.get("value") or "").strip()
            if name:
                parts.append(f"{name}={value}")
        return "; ".join(parts)

    def _extract_cookie(self, cookies, cookie_name: str) -> str:
        for cookie in cookies:
            if cookie.get("name") == cookie_name:
                return str(cookie.get("value") or "")
        return ""

    def _click_primary(self, page) -> None:
        selectors = [
            "button[type='submit']",
            "[data-testid='continue-button']",
            "button:has-text('Continue')",
            "button:has-text('Next')",
            "button:has-text('Verify')",
            "button:has-text('Submit')",
            "button:has-text('继续')",
            "button:has-text('下一步')",
            "button:has-text('验证')",
            "button:has-text('登录')",
        ]
        if self._click_first_visible(page, selectors):
            return
        raise RuntimeError("Could not find a primary action button in the browser flow")

    def _click_first_visible(self, page, selectors) -> bool:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() and locator.first.is_visible():
                locator.first.click()
                return True
        return False

    def _fill_first(self, page, selectors, value: str) -> None:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() and locator.first.is_visible():
                locator.first.fill(value)
                return
        raise RuntimeError(f"Could not find an input for selectors: {selectors}")

    def _fill_if_visible(self, page, selectors, value: str) -> bool:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() and locator.first.is_visible():
                locator.first.fill(value)
                return True
        return False

    def _set_input_value(self, page, selectors, value: str) -> bool:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count():
                locator.first.evaluate(
                    """(el, nextValue) => {
                        el.value = nextValue;
                        el.setAttribute('value', nextValue);
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    value,
                )
                return True
        return False

    def _select_if_visible(self, page, selectors, value: str) -> bool:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() and locator.first.is_visible():
                locator.first.select_option(value=value)
                return True
        return False

    def _is_visible(self, page, selectors) -> bool:
        for selector in selectors:
            locator = page.locator(selector)
            if locator.count() and locator.first.is_visible():
                return True
        return False

    def _email_selectors(self):
        return [
            "input[type='email']",
            "input[name='username']",
            "input[autocomplete='username']",
            "input[inputmode='email']",
            "input[name='email']",
        ]

    def _password_selectors(self):
        return [
            "input[type='password']",
            "input[name='password']",
            "input[autocomplete='new-password']",
            "input[autocomplete='current-password']",
        ]

    def _otp_selectors(self):
        return [
            "input[autocomplete='one-time-code']",
            "input[inputmode='numeric']",
            "input[type='tel']",
            "input[type='number']",
        ]

    def _profile_selectors(self):
        return [
            "input[name='name']",
            "input[autocomplete='given-name']",
            "input[placeholder*='name']",
        ]

    def _birthday_segment_selectors(self, segment: str):
        return [
            f"[role='spinbutton'][data-type='{segment}']",
        ]

    def _log(self, message: str) -> None:
        self.logger_callback(message)
