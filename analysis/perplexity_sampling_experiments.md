# Perplexity vs Sampling Steps Experiments

记录在不同采样步数下（`interpolant.sampling.num_timesteps`）模型推理行为与困惑度变化情况。建议保持除步数外的配置完全一致，方便对比。

## 实验环境
- 模型权重：`/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__Encoder11atoms_chroma_SNR1_linearBridge/2025-10-16_21-45-09/last.ckpt`
- 推理脚本：`python experiments/inference.py`
- 基础配置：`configs/Infer_SH.yaml`

## 配置修改说明
在 `Infer_SH.yaml` 或命令行 override 中调整以下字段：

- `interpolant.sampling.num_timesteps`: 采样步数（例如 1、10、100）。
- （可选）`experiment.inference_dir`: 每次实验换一个输出目录，避免覆盖结果。

示例命令：

```bash
python experiments/inference.py \
  experiment.ckpt_path=/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__Encoder11atoms_chroma_SNR1_linearBridge/2025-10-16_21-45-09/last.ckpt \
  interpolant.sampling.num_timesteps=10 \
  experiment.inference_dir=outputs/inference_steps10
```

## 结果记录
使用下表记录每次推理的核心指标（可从日志或 `.pdb` 输出中整理）。

| 采样步数 | 推理输出目录           | 平均困惑度 | Recovery / Top-1 | 备注 |
|----------|------------------------|-------------|------------------|------|
| 1        | outputs/inference      | 1.84        | 0.739            | 新一轮 step=1 推理，perplexity 接近 1.8，recovery ≈ 0.74 |
| 10       | outputs/inference      | 654.23      | 0.062            | 上一次 step=10 推理（覆盖同目录） |
| 100      |                        |             |                  |      |

> 可根据需要在表中继续添加更多采样步数或其他指标（例如平均侧链 RMSD）。

## 观察与结论
- 步数对困惑度的影响：
- 对结构质量的影响：
- 推荐的默认设置：

（实验完成后在此更新总结）


### Perplexity list (num_timesteps=1)
- sample_T1104-D1_000034: perplexity=626.200
- sample_T1106s1-D1_000043: perplexity=854.459
- sample_T1106s2-D1_000037: perplexity=283.305
- sample_T1109-D1_000019: perplexity=949.667
- sample_T1119-D1_000044: perplexity=696.499
- sample_T1120-D1_000033: perplexity=735.780
- sample_T1120-D2_000038: perplexity=792.254
- sample_T1121-D1_000021: perplexity=391.218
- sample_T1121-D2_000023: perplexity=317.668
- sample_T1123-D1_000020: perplexity=355.793
- sample_T1124-D1_000006: perplexity=540.208
- sample_T1129s2-D1_000002: perplexity=551.697
- sample_T1133-D1_000004: perplexity=759.706
- sample_T1137s1-D1_000028: perplexity=1219.593
- sample_T1137s1-D2_000013: perplexity=425.736
- sample_T1137s2-D1_000029: perplexity=507.571
- sample_T1137s2-D2_000022: perplexity=883.819
- sample_T1137s3-D1_000030: perplexity=875.713
- sample_T1137s3-D2_000025: perplexity=916.955
- sample_T1137s4-D1_000035: perplexity=831.550
- sample_T1137s4-D2_000015: perplexity=587.857
- sample_T1137s4-D3_000041: perplexity=378.722
- sample_T1137s5-D1_000032: perplexity=486.766
- sample_T1137s5-D2_000017: perplexity=678.346
- sample_T1137s6-D1_000027: perplexity=702.882
- sample_T1137s6-D2_000012: perplexity=664.001
- sample_T1137s7-D1_000008: perplexity=486.251
- sample_T1137s8-D1_000011: perplexity=1688.733
- sample_T1137s9-D1_000010: perplexity=1122.717
- sample_T1139-D1_000009: perplexity=750.583
- sample_T1145-D1_000040: perplexity=328.440
- sample_T1145-D2_000003: perplexity=636.485
- sample_T1150-D1_000007: perplexity=360.484
- sample_T1157s1-D1_000000: perplexity=372.017
- sample_T1157s1-D2_000036: perplexity=579.398
- sample_T1157s1-D3_000016: perplexity=716.024
- sample_T1157s2-D1_000039: perplexity=692.636
- sample_T1157s2-D2_000018: perplexity=788.387
- sample_T1157s2-D3_000031: perplexity=789.905
- sample_T1170-D1_000014: perplexity=502.108
- sample_T1170-D2_000042: perplexity=500.691
- sample_T1180-D1_000005: perplexity=565.650
- sample_T1187-D1_000024: perplexity=253.768
- sample_T1188-D1_000001: perplexity=413.815
- sample_T1194-D1_000026: perplexity=878.305
- 平均 perplexity: 654.230

### Perplexity list (num_timesteps=10, latest run)
- sample_T1104-D1_000034: recovery=0.051, perplexity=626.200
- sample_T1106s1-D1_000043: recovery=0.056, perplexity=854.459
- sample_T1106s2-D1_000037: recovery=0.054, perplexity=283.305
- sample_T1109-D1_000019: recovery=0.061, perplexity=949.667
- sample_T1119-D1_000044: recovery=0.062, perplexity=696.499
- sample_T1120-D1_000033: recovery=0.093, perplexity=735.780
- sample_T1120-D2_000038: recovery=0.109, perplexity=792.254
- sample_T1121-D1_000021: recovery=0.097, perplexity=391.218
- sample_T1121-D2_000023: recovery=0.095, perplexity=317.668
- sample_T1123-D1_000020: recovery=0.089, perplexity=355.793
- sample_T1124-D1_000006: recovery=0.063, perplexity=540.208
- sample_T1129s2-D1_000002: recovery=0.067, perplexity=551.697
- sample_T1133-D1_000004: recovery=0.049, perplexity=759.706
- sample_T1137s1-D1_000028: recovery=0.033, perplexity=1219.593
- sample_T1137s1-D2_000013: recovery=0.054, perplexity=425.736
- sample_T1137s2-D1_000029: recovery=0.081, perplexity=507.571
- sample_T1137s2-D2_000022: recovery=0.021, perplexity=883.819
- sample_T1137s3-D1_000030: recovery=0.047, perplexity=875.713
- sample_T1137s3-D2_000025: recovery=0.018, perplexity=916.955
- sample_T1137s4-D1_000035: recovery=0.052, perplexity=831.550
- sample_T1137s4-D2_000015: recovery=0.026, perplexity=587.857
- sample_T1137s4-D3_000041: recovery=0.081, perplexity=378.722
- sample_T1137s5-D1_000032: recovery=0.080, perplexity=486.766
- sample_T1137s5-D2_000017: recovery=0.036, perplexity=678.346
- sample_T1137s6-D1_000027: recovery=0.053, perplexity=702.882
- sample_T1137s6-D2_000012: recovery=0.040, perplexity=664.001
- sample_T1137s7-D1_000008: recovery=0.050, perplexity=486.251
- sample_T1137s8-D1_000011: recovery=0.012, perplexity=1688.733
- sample_T1137s9-D1_000010: recovery=0.034, perplexity=1122.717
- sample_T1139-D1_000009: recovery=0.041, perplexity=750.583
- sample_T1145-D1_000040: recovery=0.101, perplexity=328.440
- sample_T1145-D2_000003: recovery=0.078, perplexity=636.485
- sample_T1150-D1_000007: recovery=0.112, perplexity=360.484
- sample_T1157s1-D1_000000: recovery=0.086, perplexity=372.017
- sample_T1157s1-D2_000036: recovery=0.044, perplexity=579.398
- sample_T1157s1-D3_000016: recovery=0.030, perplexity=716.024
- sample_T1157s2-D1_000039: recovery=0.057, perplexity=692.636
- sample_T1157s2-D2_000018: recovery=0.078, perplexity=788.387
- sample_T1157s2-D3_000031: recovery=0.057, perplexity=789.905
- sample_T1170-D1_000014: recovery=0.062, perplexity=502.108
- sample_T1170-D2_000042: recovery=0.069, perplexity=500.691
- sample_T1180-D1_000005: recovery=0.063, perplexity=565.650
- sample_T1187-D1_000024: recovery=0.104, perplexity=253.768
- sample_T1188-D1_000001: recovery=0.089, perplexity=413.815
- sample_T1194-D1_000026: recovery=0.056, perplexity=878.305
- 平均 recovery: 0.062
- 平均 perplexity: 654.230

### Perplexity list (num_timesteps=1, latest run)
- sample_T1104-D1_000034: recovery=0.778, perplexity=1.709
- sample_T1106s1-D1_000043: recovery=0.676, perplexity=2.069
- sample_T1106s2-D1_000037: recovery=0.694, perplexity=1.945
- sample_T1109-D1_000019: recovery=0.790, perplexity=1.715
- sample_T1119-D1_000044: recovery=0.542, perplexity=2.867
- sample_T1120-D1_000033: recovery=0.797, perplexity=1.552
- sample_T1120-D2_000038: recovery=0.773, perplexity=1.983
- sample_T1121-D1_000021: recovery=0.760, perplexity=1.775
- sample_T1121-D2_000023: recovery=0.780, perplexity=1.703
- sample_T1123-D1_000020: recovery=0.762, perplexity=1.724
- sample_T1124-D1_000006: recovery=0.807, perplexity=1.625
- sample_T1129s2-D1_000002: recovery=0.751, perplexity=1.738
- sample_T1133-D1_000004: recovery=0.724, perplexity=1.931
- sample_T1137s1-D1_000028: recovery=0.740, perplexity=1.750
- sample_T1137s1-D2_000013: recovery=0.729, perplexity=2.059
- sample_T1137s2-D1_000029: recovery=0.752, perplexity=1.729
- sample_T1137s2-D2_000022: recovery=0.608, perplexity=2.088
- sample_T1137s3-D1_000030: recovery=0.718, perplexity=1.796
- sample_T1137s3-D2_000025: recovery=0.646, perplexity=1.999
- sample_T1137s4-D1_000035: recovery=0.698, perplexity=1.901
- sample_T1137s4-D2_000015: recovery=0.711, perplexity=2.012
- sample_T1137s4-D3_000041: recovery=0.730, perplexity=1.742
- sample_T1137s5-D1_000032: recovery=0.788, perplexity=1.638
- sample_T1137s5-D2_000017: recovery=0.715, perplexity=1.901
- sample_T1137s6-D1_000027: recovery=0.755, perplexity=1.690
- sample_T1137s6-D2_000012: recovery=0.694, perplexity=1.897
- sample_T1137s7-D1_000008: recovery=0.757, perplexity=1.750
- sample_T1137s8-D1_000011: recovery=0.757, perplexity=1.688
- sample_T1137s9-D1_000010: recovery=0.789, perplexity=1.720
- sample_T1139-D1_000009: recovery=0.702, perplexity=2.009
- sample_T1145-D1_000040: recovery=0.758, perplexity=1.709
- sample_T1145-D2_000003: recovery=0.779, perplexity=1.665
- sample_T1150-D1_000007: recovery=0.797, perplexity=1.636
- sample_T1157s1-D1_000000: recovery=0.800, perplexity=1.644
- sample_T1157s1-D2_000036: recovery=0.693, perplexity=2.147
- sample_T1157s1-D3_000016: recovery=0.648, perplexity=2.201
- sample_T1157s2-D1_000039: recovery=0.755, perplexity=1.730
- sample_T1157s2-D2_000018: recovery=0.816, perplexity=1.564
- sample_T1157s2-D3_000031: recovery=0.837, perplexity=1.699
- sample_T1170-D1_000014: recovery=0.775, perplexity=1.748
- sample_T1170-D2_000042: recovery=0.708, perplexity=1.927
- sample_T1180-D1_000005: recovery=0.732, perplexity=1.959
- sample_T1187-D1_000024: recovery=0.732, perplexity=1.995
- sample_T1188-D1_000001: recovery=0.778, perplexity=1.726
- sample_T1194-D1_000026: recovery=0.733, perplexity=1.759
- 平均 recovery: 0.739
- 平均 perplexity: 1.840
