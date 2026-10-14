# v0.73 参赛源码发布

| 项目 | 记录 |
| --- | --- |
| 队伍 | 0516 |
| 平台版本名 | `0516v73` |
| 本地版本 | v0.73 |
| 原包生成 | 2026-10-07 |
| 平台确认 | 2026-10-07 23:44，中国标准时间 |
| 最终版本选择 | 2026-10-08 08:58，中国标准时间 |
| 公开整理 | 2026-10-14 |
| 原始包文件名 | `observer-agent-v0.73-request-head-experimental.zip` |

发布从原始提交包提取 15 个 Python 运行文件、`observer.project.json`、`requirements.txt` 和 `LICENSE.md`。这些文件保留原始内容，逐项 SHA-256 记录于 [source-manifest.json](source-manifest.json)。

原包中的 README 和开发变更日志由本仓库的项目介绍、算法说明及本页替代。新增内容包括架构图、合成协议示例、轻量协议检查和版本来源文件。

参赛版的模型服务通过环境变量配置，调度器从协议接收星表、台址、光纤布局、夜历、公告、请求和反馈。复现时，可先运行 `python3 -B scripts/smoke.py`，再使用官方 runner 评测任务卡。

原包 SHA-256：`7b813070106da3e9a8e18f5d51821bf56e5746b0eb8ba2df37cf1a2bd34cc853`。
