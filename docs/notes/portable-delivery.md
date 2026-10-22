# 从可运行代码到独立交付

仅在 checkout 外启动 wheel 不足以证明安装可脱离源码：原检查器仍借助项目的 uv.lock 同步环境。
最终交付套件包含 wheel、冻结 Decoder、PEP 751 pylock、标准库安装脚本和逐文件清单。
pylock 从已有 uv.lock 导出，保留精确版本、平台条件、工件 URL 和 SHA256，不重新选择依赖。

安装脚本无需导入 Kaleido。先验证清单，再在全新 venv 安装锁定运行时与普通 wheel，
最后以 python -I 校验安装位置、HF 包缺席及已验证样例的逐token输出。已有环境拒绝覆盖，
失败环境保留用于诊断。首次可下载依赖；离线模式要求已准备的 uv cache，可显式指定 cache-dir。
套件不内嵌大体积依赖 wheel，也不包含 Qwen 或语料。

CI 把套件复制到临时目录，以基础 Python 启动独立安装脚本，工作目录不在源码中。
通过后上传可下载 ZIP、安装重放报告和外部摘要，便于合并前审阅。
main 的手动发布流程重跑同样检查后只创建 draft release，不自动公开、不覆盖既有 tag。

参考：[uv 的 pylock 导出](https://docs.astral.sh/uv/concepts/projects/export/)、
[pylock 安装](https://docs.astral.sh/uv/pip/compile/)。
