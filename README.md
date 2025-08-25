intraday-volume-ml/
├─ README.md
├─ requirements.txt              # 依赖（或用 conda/pyproject 自行管理）
├─ .gitignore
├─ configs/
│  └─ default.yaml               # 全局配置（数据/模型/训练/评估/路径）
├─ data/                         # 放原始/中间数据（本仓库可不跟踪）
│  ├─ raw/
│  └─ interim/
├─ outputs/
│  ├─ checkpoints/               # 训练中保存的模型权重
│  ├─ metrics/                   # 评估结果(JSON/CSV)
│  └─ logs/                      # 训练日志
├─ notebooks/
│  └─ 00_eda.ipynb               # 可选：探索性分析
├─ src/
│  ├─ __init__.py
│  ├─ config.py                  # 读取/校验 YAML 配置
│  ├─ train.py                   # 训练入口（含早停、调度、保存最优）
│  ├─ eval.py                    # 评估入口（inner_test & outer_test）
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ uam_threeway_prep.py    # ★ 数据加载/切分/窗口采样（dense/sparse）
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ deeplobv.py             # ★ DeepLOBv 模型（Causal Conv + Inception + LSTM）
│  └─ utils/
│     ├─ __init__.py
│     ├─ metrics.py              # masked MSE/MAE/R2
│     ├─ common.py               # set_seed、save/load json、设备选择
│     └─ logging.py              # 简单日志封装
└─ scripts/
   ├─ run_train.sh               # 一键训练（可选）
   └─ run_eval.sh                # 一键评估（可选）
