# SGLang 0.5.16 接入核对

此页依据服务器预检的包版本，以及镜像来源说明中的源码 commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`。以下为该 commit 的静态源码结果；镜像内源码是否包含额外补丁仍须对照实际文件哈希。

## MoE 接入边界

- [deepseek_v4.py](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/deepseek_v4.py#L1323) 实例化 `deepseek_v2.DeepseekV2MoE`。adapter 应在 routed-expert 执行边界接入，保留 attention、mHC、共享专家和缓存语义。
- [deepseek_v2.py](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/models/deepseek_v2.py#L591) 使用 `num_hash_layers` 区分早期 HashTopK；需保留 input IDs，不能用普通 top-k 实现替换所有层。原生 `topk_output` 是路由参考的输入。
- `routed_scaling_factor` 在 top-k、runner、外层合并之间由运行时能力标志决定。CPU `combine_results` 的输入权重必须先被 adapter 转成最终权重；它将权重乘一次，共享专家另由原生 runtime 加入。需逐层验证，不能在顶层再无条件乘一次 1.5。

## 原生 EP4 候选及限制

[FusedMoE](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L224) 按 EP size 计算本地专家数，并按 MoE TP size 划分 intermediate dimension。

[StandardDispatcher](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/moe/token_dispatcher/standard.py#L170) 存在 EP>1 的 global-to-local 专家映射，非本地专家 ID 变为 -1。外层 `forward_normal` 存在结果 all-reduce。因此 `moe-a2a-backend=none` 表示没有选择专用 all-to-all backend，不能单独证明专家没有分片。

[Fp8Config](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/fp8.py#L371) 对原版 MXFP4 专家与 Marlin runner 分派到 `Mxfp4MarlinMoEMethod`，保留其余模型部件的混合精度。该方法处理 E8M0 scales、Marlin weight repacking，并把 w1/w3 保持为 gate/up 顺序。

**必须验证的具体风险：**[Mxfp4MarlinMoEMethod.apply](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py#L142) 构建的 quant info 未显式设置 expert_map；[fused_marlin_moe](https://github.com/sgl-project/sglang/blob/fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1/python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py#L276) 的 EP 开关和部分中间缓冲清零依赖 expert_map。需要实际验证映射后的 -1 slot 是否正确归零，不能仅凭服务 ready 判定 EP 数值正确。

## 本轮可执行入口

`scripts/run_native_ep.py` 默认只输出 JSON 命令预览，不访问 Docker。GPUQ 已分配 GPU 4–7 后，在宿主运行：

```bash
python3 scripts/run_native_ep.py --work-root /实际可写工作盘路径
python3 scripts/run_native_ep.py --work-root /实际可写工作盘路径 --execute
```

候选为 TP=4、EP=4、Marlin、standard/none dispatcher，默认端口 30001、concurrency=1，原版模型只读挂载。脚本执行前把镜像 tag 解析成 image ID，实际启动用该 ID；命令、GPUQ job ID、退出码和服务日志保存在 WORK_ROOT 下的独立目录。需由服务器 GPUQ 作业管理其运行和终止。

此入口用于原生服务启动核查，不是正式计时器。`generation_status` 与 `ep_correctness_status` 固定为未验证，性能为未运行。服务启动后仍需检查实际 EP size、每 rank 64 个 routed experts、零贡献槽、共享专家加和、短生成和 graph replay；通过后才进入每组 1200 秒的正式对照。

## CPU 判据

`src/weave/routing.py` 保留每个 origin token 的唯一身份、逐目标 rank 的去重 payload 和逐专家 top-k slot。combine 拒绝缺失/重复/未知 slot，并保留原生路由缩放。此模块是 CUDA 实现的结果判据，不承载 GPU 通信或专家矩阵乘。
