# GPU编号由调度器分配

用户已取消固定物理GPU4–7的限制。当前启动入口读取GPUQ设置的 `CUDA_VISIBLE_DEVICES`，接受任意四个不同的非负物理GPU编号，例如`0,1,2,3`或`1,3,5,7`。当前模型实验仍要求四张H100、TP4/EP4。

宿主入口共用 `src/weave/gpu_allocation.py`：CUDA core、preflight、校准、模型验收、完整实验及native/Weave服务。Docker device参数使用实际分配并保留CSV内层引号；容器可见ordinal由选中卡数派生，四卡时为0..3。CPP的rank/ordinal绑定是容器内逻辑编号。

分配缺失、重复、负编号或不是四张卡时在启动前报错。GPUQ实际分配后，直接执行现有入口：

```bash
python3 scripts/run_full_experiment.py \
  --work-root /srv/weave-work \
  --export-root /srv/weave-work/public-reports \
  --records /home/jovyan/wangtonghan/weave-deepseek-v4-flash/data/gpqa/prepared/deepseek-v4-flash/main.jsonl \
  --core-run-dir /srv/weave-work/cuda-functional.047ed4696f52 \
  --calibration-dir /srv/weave-work/calibration.047ed4696f52 \
  --calibration-kernel-timeout-seconds 120
```

CLI仅预览时可以用下例查看设备参数；执行GPU任务使用调度器实际授权的分配：

```bash
CUDA_VISIBLE_DEVICES=1,3,5,7 python3 scripts/run_native_ep.py
```

报告的 `physical_gpus`/`host_gpu_allocation`、Docker的 `WEAVE_HOST_ALLOCATION` 和nvidia-smi快照均记录本次选择。编号列表表示物理选择，不应仅靠列表顺序推断rank与PCI设备的对应关系。模型验收与正式两组服务会检查启动报告的选择与当前作业一致。

历史报告中的4–7保留原始含义。CUDA源码digest未改变；旧core/校准复用执行已有源码、checkpoint、镜像、rank、Hopper SM90和profile设备身份校验，新分配的IPC/设备可用性由实际运行检查。

服务器继续遵守AGENTS.md的GitHub取材和离线运行约束。
