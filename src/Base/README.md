# Base

Base 目录保存非 PTL 的基线模型、预训练/微调流程、对比实验和分析脚本。文件已按用途归类：

- `models/`: 只包含模型结构定义，例如 MLP、CNN、LSTM、CLA、普通 Transformer、direct comparison baseline。
- `training/`: Base 预训练和微调入口脚本，包括原始版与 optimized 版本。
- `benchmarks/`: 日级 benchmark 和 fair-compare 实验入口。
- `analysis/`: 结果分析和画图脚本。
- `legacy/`: 历史实验脚本，保留以便复现旧流程。

常用入口示例：

```bash
python src/Base/training/pretrain.py
python src/Base/training/finetune_optimized.py
python src/Base/benchmarks/benchmark_daily_mlp.py
python src/Base/benchmarks/benchmark_daily_bilstm.py
python src/Base/benchmarks/benchmark_daily_yangshuo_nh4n.py
```

脚本会根据自身路径定位仓库根目录，默认数据和结果路径仍指向仓库下的 `data/` 与 `results/`。
