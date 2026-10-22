请根据附件中的实施计划，开始实现原生 SGLang 的 MoE tail 通信与计算调度执行器。

请完成实际代码、必要测试、服务器执行入口和 GitHub 交付，不要只返回设计方案。已有授权内的常规修改、测试和提交无需重复询问。

一、先阅读并固定依据

工作目录：
/Users/a1234/Documents/ChatGPT/Weave

GitHub：
https://github.com/rh2844417786/weave-deepseek-v4-flash

实施计划附件：稍后补充

请先：
- 检查 git status，读取适用的 AGENTS.md。
- 安全同步最新 main，记录实际 commit SHA。
- 保留本地修改、缓存和历史报告，不使用 reset --hard 或 git clean。
- 阅读附件，以及 docs/NATIVE_DYNAMIC_BUDGET.md、docs/NATIVE_COST_MODEL.md、现有 native_budget/native_cost 源码、固定 Marlin 来源和已有服务器证据。
- 不要假设附件记录的 33d0719 仍是最新版本。
- 将附件作为设计依据，核对其中的源码判断和接口；不要机械照抄尚未验证的伪代码。

二、目标和方法边界

目标仍是在原生 SGLang 上提高 DeepSeek-V4-Flash 的模型吞吐。batch 32 是初始对照变量。

本轮实现的准确范围是：

保留原生路由、grouping、Marlin gate/up、activation、shared experts 和 prefill；
重构 decode 的 down projection、本地结果合并和输出归约；
在同一个设备端 resident executor 中，实现真实的通信/计算角色、任务领取、提前交付和通信 worker 窃取 down 任务；
最后加入实测成本模型与 GPU 计划选择。

采用附件推荐的 A 路线：
保留复制输入和本地专家计算，不新增 dispatch all-to-all。

本方法建议使用独立身份：
native_marlin_peer_tail_v1

这是 MoE 后半段的 Weave 机制适配，不是论文五阶段完整复现。保留旧 native_budget、native_cost 的身份和历史结果，不覆盖它们。

不要把下列内容作为本轮目标的替代交付：
- 再次调整两个 GEMM 的 CTA 比例；
- 给旧成本表增加更多特征；
- host 依次启动计算和通信 kernel，却称其为 worker stealing；
- 仅有 enabled 标志，没有真实任务执行证据；
- 将有效 CTA 数直接解释成独占物理 SM 数。

三、必须保持的计算与集成语义

1. 保留原始混合精度 checkpoint、原生权重重排布局、反量化及 Tensor Core 算术。
   不改成另一套 FP8 激活路线，不退回旧功能性专家算子。

2. 允许重构 down 的执行组织。
   不能要求它继续是完全不透明的独立 Marlin kernel，同时又宣称通信 worker 能直接执行它的 tile。

3. 从原生 front 拆出 DownInputs：
   activation、grouping、slot 映射、top-k 权重、w2/scales 及实际计算几何。
   保证 Tensor 生命周期和地址稳定，不能先完整执行原生 MoE，再额外执行新 down。

4. 严格沿用当前原生 routing weight、routed scale 和 shared 加法语义。
   防止重复缩放、重复 shared 加法及重复 all-reduce。
   padding、远端屏蔽专家和空本地工作必须正确处理。

5. 优化路径只替换本层 decode MoE tail 的归约。
   prefill、attention 和其他 collective 保持原有路径。
   使用明确的 decode helper，不要全局 monkey patch all-reduce 为 no-op。

6. 附件提出在 tail 入口前等待 shared。
   请明确实现该依赖，并单独记录前移 join 对原有重叠的影响。
   未验证其他 shared 布局时，应明确报告不支持，不套用错误的加法位置。

四、按 S1 到 S7 实施

先形成 S1、S2 的独立代码和服务器检查入口，再连接后续阶段。
在具备源码依据时继续推进实现；GPU 依赖事项必须由服务器验证，不能用 CPU 测试代替。

S1：提取可调用的 Marlin down 计算单元

- 从固定版本 Marlin 提取加载、反量化、fragment、MMA、CTA 内归约和写回。
- 提供全 CTA 一致调用的 device helper，而不是启动子 kernel。
- 第一版采用一个原生 expert group、一个输出 N tile、完整 K 的独立任务。
- 保留原生 M 分组及适用 specialization。
- 移除对原 grid stripe 和跨 CTA split-K 锁伙伴的依赖。
- 每项任务结束时，正确排空异步加载并清理可复用 scratch 状态。
- packed 权重与 scale 的通道偏移必须按原布局寻址。

先用普通静态映射测试 kernel 调用该 helper，不加入队列或通信。

交付真实 checkpoint 的 GPU 回放入口，记录：
计算覆盖、有限数值差异、非有限输出、内存检查、资源占用和耗时。
分别报告原生 down 与提取版本的时间，不能默认提取后仍保留原生速度。

S2：实现独立的设备端分片归约

- 使用现有 CUDA IPC/peer 能力，不新增在线依赖。
- 第一版 C=1，不带 Marlin 计算。
- 实现四 owner 分片、读取 peer partial、owner 求和、读取归约结果。
- 每个 rank 写自己的 workspace，其他 rank 在 ready 后读取。
- 定义 session、invocation、epoch、buffer slot、ready、done、ack 和 abort。
- 防止快 rank 覆盖慢 rank 尚未消费的数据。
- 正确处理 device/system scope 的发布、可见性和读取。
- 不用 volatile 或 leader 的单次 fence 代替完整同步协议。

先用预生成 local partial 检查协议，并在所有输入已就绪时对照原生 all-reduce，分离通信自身开销。

同时定点验证：
四卡 peer IPC、所需同步能力，以及所选 resident/cooperative launch 的 CUDA Graph capture/replay。

根据最终编译 kernel 的实际资源占用确定可驻留 grid。
不得使用未经证明能够前进的 oversubscribed 自旋方案。

S3：组合为统一 resident executor

- 同一个 kernel 中包含计算和通信能力。
- 固定角色，先 C=1、stealing=0。
- 计算 worker 领取 down/local-finalize 任务。
- 通信 worker 执行真实 peer 通信和归约任务。
- 使用有界描述符、原子领取和明确依赖状态。
- 不允许 worker 领取未就绪任务后无限占住资源等待。
- 验证任务集合恰好执行一次，不能只比较总完成数。
- 空本地专家工作仍须参与通信协议。

记录同一 invocation 中实际执行的任务类型、worker 角色和 SM 分布。
此时仍称 worker 配额，不能直接宣称精确物理 SM 分区。

S4：加入提前交付，接入原生 decode

- 沿 down 输出通道分块，先 C=2，随后再开放有依据的 C=4。
- 保留专家输入行分组，不因 chunk 切小 M。
- 某 chunk 的全部本地贡献、local finalize 和相应 shared partial 完成后，才能发布 ready。
- 明确通信布局及四 owner 分片对齐。
- 让早期 chunk 的真实通信与后续 chunk 的 down 计算重叠。
- 接入原生 decode helper，prefill 仍使用原始函数。
- 保留 hash/top-k 所需 input_ids 等原有参数。
- 更新 coverage，检查一次原生 gate/up 和一次 tail，不能沿用“两次原生 Marlin”的旧检查。

验证同一卡时间线上实际发生的计算/通信重叠。
跨卡用 invocation 和 ready 依赖关联，不能直接相减不同 GPU 的绝对时钟。

S5：实现真实 stealing

- 通信 worker 在初始通信未就绪窗口，从同一个 down 队列领取计算任务。
- 使用与计算 worker 相同的 device helper。
- 每完成一个 tile 检查通信状态，采用有限 stealing 配额。
- 保留必要的通信进度 worker；单通信 worker 时先禁用 stealing。
- 任务执行完成、异步流水排空后才能切换角色。
- 计算结束后允许计算 worker 加入剩余通信任务。

证据必须包含：
初始 COMM 角色实际领取并完成 down task，
随后恢复通信，以及通信 ready-to-start 延迟。

受控机制测试与自然服务观察分别报告。
自然服务 stealing 为零时如实记录。

S6：建立新成本模型和 GPU 联合选择

- 使用独立的新表和策略身份，不复用旧 CTA 表中的耗时。
- 搜索执行器真正支持的通信 worker 数、输出 chunk 数、stealing 配额及 serial 模式。
- 固定 captured worker grid、threads 和 scratch 几何，不能把这些伪装成任意可变参数。
- 模型覆盖 down、local finalize、peer 通信、并发干扰、分块损失、角色切换和协议结束确认。
- 将保留的前半段、shared join 变化、工作量统计和 planner 开销纳入完整路径。
- 不把 gate/up 的 FLOPs 算成可被此 tail 执行器窃取的工作。
- 使用实际分块依赖预测完成时间，不直接套未经验证的重叠公式。

各 rank 可以采用不同角色配额，但 invocation、chunk 数、dtype 和通信布局必须一致。
计划交换和搜索在 GPU 上完成，避免逐 token CPU 读回及重新捕获。

允许选择不分块、不 stealing、serial 或始终同一个计划。
缺少匹配实测表时给出明确错误及生成命令，不伪造耗时、不静默回退。

S7：统一服务器流程与正式对照

提供统一入口，按依赖完成：
能力检查、计算检查、通信检查、统一执行器检查、必要校准、真实服务对照。

正式只运行：
- native：原始 SGLang；
- peer_tail：原生前半段加新 tail。

两组均 full CUDA Graph，初始并发 32，各完整测量 1200 秒。
沿用自然题长、自然 EOS、Think High 和当前生成配置。
不通过重复题目、强行等长输出或提前结束窗口制造收益。

正式窗口关闭阶段事件和逐任务时间线。
协议、统计、搜索和控制消息等真实运行成本必须保留。
边界读取聚合证据，不在窗口内同步回读诊断状态。

五、文件组织与来源

可参考附件建议：
- experimental/native_tail/
- src/weave/native_tail*.py
- configs/native_tail_sources.json
- docs/NATIVE_TAIL_EXECUTOR.md
- scripts/run_native_tail.py
- tests/gpu/check_native_tail_*.py

文件名可按实际仓库结构调整，但职责和方法身份必须清楚。
采用私有 overlay 和独立 JIT/cache 身份，不修改镜像内安装源码。
核验所有实际复用或修改的上游来源。

六、验证政策

所有当前实验的有限数值差异、生成差异、预测误差和性能下降均为 record_only。
不得新增 relative-L2、absolute error、allclose 或替代门限。
不得恢复跨候选 routing digest 或工作量特征一致性门限。

保留：
非有限输出、非法访存、计算/任务覆盖、同步和协议有效性、
编译运行、来源身份、graph replay、取消超时及清理检查。

详细任务位图和时间线只用于独立验证。
正式运行不长期保留原始 grouping 或逐任务诊断数据。

性能下降要分清：
down 提取、取消 split-K、队列、统一编译资源、
shared join 前移、新通信、分块以及 planner 的成本。
不因吞吐下降将有效实验标为数值失败。

七、服务器约束

- 服务器只能从 GitHub 获取材料。
- 使用已有 lmsysorg/sglang:v0.5.17 镜像及现有安装包，重新核对运行契约。
- 原始模型只读使用：
  /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash
- 工作根目录：
  /srv/weave-work
- 使用 GPUQ 实际分配，不硬编码物理卡号。
- 不在线 pip、apt、HF 下载，不拉镜像。
- 新外部材料确有必要时，在联网开发机准备 Linux/CUDA 兼容、版本固定且有摘要校验的 GitHub 交付。
- 保留缓存和历史报告。
- 原始题目、路由、激活、权重、生成内容和完整日志留在服务器私有目录。
- 只公开白名单聚合证据及 SHA256 manifest。

八、本轮交付要求

先把 S1、S2 做成独立可检查的提交和服务器命令，再按依赖推进后续实现。

本机没有 GPU 时：
- 完成有依据的源码、CPU 测试和 GPU harness；
- 生成服务器可离线执行的统一入口与完整 prompt；
- 明确哪些事项等待真实 GPU 验证；
- 不声称 CUDA 编译、机制执行或吞吐已经成功。

完成后提交并推送 GitHub，服务器 prompt 写明实际分支及执行 commit，保留已有未提交修改，禁止强推覆盖。

最终汇报：
1. 实际实现到哪个阶段，修改了哪些源码；
2. 本地测试结果与适用边界；
3. GitHub 提交和来源身份；
4. 服务器统一执行命令；
5. 可直接交给服务器 Claude Code＋DeepSeek V4 Flash 的完整 prompt；
6. 尚须 GPU 实验确认的事项。

分别报告：
source_implemented
gpu_mechanisms_verified
full_model_graph_execution_verified
throughput_result_observed

不要用一个“实现成功”混合这四个完成层级，也不要将本方法标为完整 Weave 论文复现。