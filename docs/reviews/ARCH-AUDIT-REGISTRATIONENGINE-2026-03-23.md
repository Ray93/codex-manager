# RegistrationEngine 深度架构审计与失败日志合并报告

日期: 2026-03-23
范围:
- 代码审计: `src/core/register.py`, `src/core/http_client.py`, `src/services/tempmail.py`, `src/web/routes/registration.py`
- 日志样本: `logs/app.log` 中最近 100 个失败任务

## 1. 执行摘要

结论分两层:

1. `RegistrationEngine` 当前是一个集中式顺序控制器，控制面、数据面、状态面耦合在单类内部，闭环存在“观测粗、控制粗、误差分类弱”的结构性问题。
2. 最近 100 个失败任务中，`Timeout` 是主导故障，`429` 是次级但高度集中，`403` 是低频单点。按失败任务计数:
   - Timeout: 44 次
   - 429: 11 次
   - 403: 1 次
   - 其他: 44 次

就统计意义而言，`Timeout` 占比 44%，95% Wilson 区间为 34.7% 到 53.8%，显著高于 `429` 的 11% 和 `403` 的 1%。这说明当前首要瓶颈不是 Cloudflare 封锁，也不是邮箱创建限流，而是 OTP/授权后半程的时滞与恢复路径失真。

## 2. CSE 闭环审计

### 2.1 控制拓扑

- Plant:
  - OpenAI 授权链路
  - 临时邮箱供应商
  - 代理网络
  - 本地数据库与任务状态
- Controller:
  - `RegistrationEngine.run()` 的顺序式流程控制
  - `src/web/routes/registration.py` 的邮箱服务切换与任务终态写回
- Sensors:
  - `_log()` 输出
  - `logs/app.log`
  - 数据库任务日志
  - `TaskManager` 内存状态
- Actuators:
  - HTTP 请求
  - 邮箱创建与轮询
  - OAuth 重入
  - 邮箱服务 failover
  - 代理选择

### 2.2 闭环优点

- 主流程步骤清晰，按阶段推进，适合做阶段化观测，入口位于 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L1015)。
- 路由层已经有邮箱服务候选集与限流熔断雏形，见 [registration.py](/Volumes/Work/code/codex-manager/src/web/routes/registration.py#L403) 和 [registration.py](/Volumes/Work/code/codex-manager/src/web/routes/registration.py#L457)。
- `Tempmail` 429 已有最小检测链路，日志可追踪到供应商限流，见 [tempmail.py](/Volumes/Work/code/codex-manager/src/services/tempmail.py#L81)。

### 2.3 闭环缺陷

#### 1. 控制器过大，导致误差无法在局部收敛

`run()` 集成了 IP 检查、邮箱供应商交互、OpenAI 授权、OTP、Workspace、OAuth 回调与结果持久化前状态填充，单方法过长，且依赖大量实例可变状态，见 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L1015)。

后果:
- 任一阶段失败都被压平为布尔值返回。
- 控制输入只能“继续/返回失败”，无法做细粒度补偿。
- 失败分类被终态字符串覆盖，真实物理故障被折叠成“获取 Workspace ID 失败”等代理错误。

#### 2. 传感器语义不足，导致 Timeout 被错误归类到 Workspace

OTP 拉取失败只返回 `None`，`run()` 再把后续失败归并到 Workspace 路径，见 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L442) 和 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L1080)。

日志证据:
- [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L54897) 到 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L54904) 显示先发生“等待验证码超时”，终态却写成“获取 Workspace ID 失败 (含降级补偿)”。

这会让控制器错误地把邮箱时滞问题当作授权后段问题处理。

#### 3. 控制面只对邮箱服务 429 做局部闭环，未覆盖代理与授权面

路由层只在 `RateLimitedEmailServiceError` 场景下切换邮箱服务，见 [registration.py](/Volumes/Work/code/codex-manager/src/web/routes/registration.py#L457)。但最近 100 个失败任务里，真正占大头的是 OTP Timeout 与 Workspace 缺失，而这两个问题都没有对应的控制输入:

- 没有代理信誉降级
- 没有 OAuth/Workspace 阶段的代理切换
- 没有 OTP 第二阶段的独立重试预算

#### 4. HTTP 客户端重试策略与故障形态不匹配

`HTTPClient.request()` 只对 `>=500` 做重试，不对 429 做退避，也不区分 403/429/401 的控制意义，见 [http_client.py](/Volumes/Work/code/codex-manager/src/core/http_client.py#L112)。

后果:
- 429 会直接回传业务层，业务层只能失败或靠外层熔断。
- 403 无法触发代理信誉降级。
- 401 无法触发登录流重建。

#### 5. 状态面与观测面有重复副作用

`_log()` 同时写内存、回调、数据库、全局日志，见 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L139)。这让传感器与状态面耦合:

- 日志故障可能反噬主流程
- 同一事件被多次展开，难以统一结构化分析
- 控制器无法只输出“事件”，必须直接决定落盘方式

## 3. Clean Code 审计

### 3.1 主要坏味道

- God Object: `RegistrationEngine` 同时承担编排器、网络客户端协调器、状态容器、日志器和部分持久化语义。
- Primitive Obsession: 大量 `bool` / `Optional[str]` 返回值承载复杂故障。
- Duplicate Logic: 登录密码提交拆成两个几乎重复的方法，见 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L747) 和 [register.py](/Volumes/Work/code/codex-manager/src/core/register.py#L793)。
- Temporal Coupling: `self.email`, `self.password`, `self.oauth_start`, `self.session`, `self._otp_sent_at` 必须按隐含顺序写入，稍有偏差就会失真。
- Error Flattening: `创建用户账户失败`、`获取 Workspace ID 失败` 等终态过于粗糙，无法直接反映物理根因。
- Mixed Concerns: 任务路由函数 `_run_sync_registration_task()` 同时做代理选择、邮箱服务选择、引擎执行、自动上传和数据库状态收口，见 [registration.py](/Volumes/Work/code/codex-manager/src/web/routes/registration.py#L362)。

### 3.2 冗余与可收敛点

- `_submit_login_password_step()` 与 `_submit_login_password_step_and_get_continue_url()` 可以合并为一个返回结构化结果的方法。
- `run()` 中多个阶段共享“发请求 -> 记录状态码 -> 解析错误 -> 决定控制动作”的模板，可抽成 phase runner。
- `_log()` 的数据库写入应从引擎剥离到事件订阅层。
- `TempmailService.get_verification_code()` 明确写明 `otp_sent_at` 暂不使用，见 [tempmail.py](/Volumes/Work/code/codex-manager/src/services/tempmail.py#L121)。这与双阶段 OTP 场景存在直接脱节。

## 4. 最近 100 个失败任务的物理分布

样本窗口:
- 起点: [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L25252)
- 终点: [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L60766)

分类结果:

| 类别 | 次数 | 占比 | 95% Wilson 区间 |
| --- | ---: | ---: | --- |
| Timeout | 44 | 44% | 34.7% - 53.8% |
| 429 | 11 | 11% | 6.3% - 18.6% |
| 403 | 1 | 1% | 0.2% - 5.4% |
| 其他 | 44 | 44% | - |

### 4.1 Timeout

核心事实:
- 44 个 Timeout 中，43 个都表现为“等待验证码超时 -> 终态记为获取 Workspace ID 失败 (含降级补偿)”。
- 代表日志见 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L54897) 到 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L54904)。

解释:
- 这不是纯 Workspace 故障，而是第二阶段 OTP 没有在邮箱侧及时可见。
- `TempmailService.get_verification_code()` 轮询固定 120 秒，且不使用 `otp_sent_at` 做新旧邮件裁剪，见 [tempmail.py](/Volumes/Work/code/codex-manager/src/services/tempmail.py#L121)。
- 因为控制器把 OTP 超时后的降级流和 Workspace 解析串在一起，最终把上游时滞扭曲成下游授权失败。

统计结论:
- Timeout 是主导根因，且占比显著高于 429。
- 从控制论角度，这是“传感器滞后 + 误差归因错误”而不是单纯接口失败。

### 4.2 429

核心事实:
- 11 个 429 全部落在 `Tempmail.lol /inbox/create`，即邮箱创建阶段。
- 代表日志见 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L52280) 到 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L52282)。

解释:
- 这是单供应商、单接口、单阶段的集中限流，不是全链路随机波动。
- `HTTPClient` 不对 429 做退避重试，见 [http_client.py](/Volumes/Work/code/codex-manager/src/core/http_client.py#L117)。
- 路由层虽然有邮箱服务熔断与切换框架，但在这些失败样本里仍然表现为直接失败，说明供应商多样性或默认候选配置仍不足。

统计结论:
- 429 是第二优先级问题。
- 其特征是“集中、可隔离、可通过供应商调度降低”。

### 4.3 403

核心事实:
- 最近 100 个失败任务里只有 1 个 403。
- 代表日志见 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L55373) 到 [app.log](/Volumes/Work/code/codex-manager/logs/app.log#L55374)。
- 响应体是 Cloudflare `Just a moment...` 页面，不是业务 JSON。

解释:
- 这是代理信誉或指纹挑战问题，不是注册表单协议错误。
- 403 是低频离群点，不能作为当前主优化方向。

统计结论:
- 403 目前不构成主导故障模式。
- 但应该进入代理评分与预检体系，避免在高价值任务上触发。

## 5. 代码收敛方案

### 5.1 第一阶段: 拆控制器，不改行为

- 将 `run()` 拆成显式 phase:
  - `ip_check`
  - `email_prepare`
  - `signup`
  - `otp_primary`
  - `account_create`
  - `oauth_reenter`
  - `otp_secondary`
  - `workspace_resolve`
  - `oauth_callback`
- 每个 phase 返回统一的 `PhaseResult`:
  - `success`
  - `phase`
  - `error_code`
  - `http_status`
  - `retryable`
  - `next_action`

目标:
- 保持现有输入输出不变。
- 先让误差可观测，再谈策略优化。

### 5.2 第二阶段: 分离控制面与执行面

- `RegistrationEngine` 只保留编排。
- HTTP 请求、OTP 拉取、Workspace 解析、OAuth 回调分别下沉为独立 executor。
- `_log()` 改成事件发布，不在引擎内部直接写数据库。

目标:
- 控制器只负责状态跃迁。
- 执行器只负责副作用。
- 观测器统一消费事件。

### 5.3 第三阶段: 建立真实失败分类

- 终态错误码至少拆出:
  - `OTP_TIMEOUT_PRIMARY`
  - `OTP_TIMEOUT_SECONDARY`
  - `EMAIL_CREATE_RATE_LIMITED`
  - `SIGNUP_FORBIDDEN_CLOUDFLARE`
  - `WORKSPACE_COOKIE_MISSING`
  - `LOGIN_PASSWORD_401`
  - `REGISTRATION_DISALLOWED`
- 路由层不要再把多种物理根因压成 `获取 Workspace ID 失败`。

### 5.4 第四阶段: 压缩重复逻辑

- 合并两个登录密码提交方法。
- 抽象“请求 + 状态码记录 + 错误解析”模板。
- 将 `_run_sync_registration_task()` 里的自动上传流程拆到 post-success hook，避免任务执行与外部同步混在一个函数里。

## 6. 针对这 100 次失败的物理优化策略

### 6.1 Timeout 优化

- 把 OTP 第二阶段单独建预算，不复用第一阶段固定 120 秒。
- `TempmailService.get_verification_code()` 使用 `otp_sent_at` 过滤旧邮件，避免第二次 OTP 被第一次邮件污染。
- 第二次 OTP 超时后，先做邮箱供应商刷新或换供应商，再做 Workspace 解析；不要直接进入 Workspace 失败终态。
- 记录每个域名、每个供应商的 OTP 到达延迟分位数，按 P50/P95 选择优先级。
- 对“OTP 二次等待”引入更短轮询间隔和更快刷新，而不是简单把总 timeout 拉长。

### 6.2 429 优化

- 为邮箱创建接口单独做 429 退避，不依赖通用 HTTP 客户端的 5xx 逻辑。
- 将 `retry_after`、冷却结束时间和供应商失败率持久化，不只保存在进程内。
- 默认配置至少准备两个可切换邮箱供应商，不让单一 `Tempmail.lol` 成为硬依赖。
- 在批量模式下给邮箱创建阶段加令牌桶，平滑 03:11 到 03:27 的创建尖峰。

### 6.3 403 优化

- 在真正启动注册前，对代理做一次低成本授权页预探测；若命中 Cloudflare challenge，直接换代理。
- 给代理建立信誉分，403 一次即降权，不再继续分配到注册主链。
- 403 不需要扩大主流程重试次数，应该做代理层淘汰。

## 7. 最终判断

本轮审计的核心判断是:

- 代码层面，真正需要收敛的不是“再补几个 if”，而是把 `RegistrationEngine` 从大一统顺序脚本收敛成阶段化控制器。
- 物理层面，最近 100 次失败的首要矛盾是 OTP/降级链路的时滞失真，其次才是邮箱创建 429，403 目前只是低频外部扰动。

如果只优化 429 或 403，而不重构 Timeout 的归因与控制输入，失败面不会明显下降。
