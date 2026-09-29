# 给 GPT-6 Sol 的编码交接

> 历史交接：以下内容针对 Qwen3.5-35B-A3B。用户现已改用原版 DeepSeek-V4-Flash 混合精度权重；请先读 `docs/DEEPSEEK_V4_FLASH_MIGRATION.md`，勿把下列 BF16、模型维度及 Qwen tokenizer 合同用于新目标。

请在本项目实现 Qwen3.5-35B-A3B BF16 的完整 Weave 复现和 SGLang 对照。先完整阅读：

1. `docs/superpowers/specs/2026-09-29-weave-qwen35-design.md`
2. `docs/superpowers/plans/2026-09-29-weave-qwen35.md`
3. `data/gpqa/README.md` 与 `data/gpqa/length_audit.json`

用户已确定：在 Docker 中运行完整模型；服务器权重 `/mnt/public_data/Qwen/Qwen3.5-35B-A3B`；4×H100，EP=4；GPQA batch=1，thinking 开启，最多生成 32K（32768）tokens；只比较 SGLang 原生和完整 Weave；每组服务ready后的预热与测量不超过1200秒。编码端不能直接连接服务器，通过GitHub同步，服务器git pull后执行并回传证据。

Weave 没有现成代码，需根据论文自行实现 routed-MoE persistent megakernel、路由统计驱动的 c/K 联合选择、chunk pipeline、bubble stealing、dispatch 去重与 combine。复用同一 SGLang runtime 处理 attention/线性attention、共享专家、缓存与采样。不能把仅修改SM配置、仅优化prefill、静默回退原生decode或空壳包装命名为完整Weave。

GPQA已下载；Main448题按当前官方模板的输入中位数213、最长2780tokens，没有8K样本。保留自然输入，不擅自拼接/填充或增加实验矩阵。速度目标是decoder实测，论文prefill收益不能当成decode结果。没有必须加速的阈值，但必须通过数值/路由/多卡/graph正确性门槛。

先完成baseline采集器，并请服务器提供实际SGLang配置；同时可继续CPU参考、数据和计时模块。实际baseline配置、镜像digest、GitHub目标URL尚未提供，不猜测。已有v0.5.19源码只作为插入点参考。每个阶段按计划交付可运行代码和准确的验证状态，远程未运行就标记not_run。

此交接时实际完成的仅有数据下载/审计脚本、元数据和设计/编码计划，没有GPU核心实现或性能结果。原始CSV与模型tokenizer本地存在但被Git忽略；服务器运行`python3 scripts/fetch_gpqa.py`可从作者源获得同一版本的数据。
