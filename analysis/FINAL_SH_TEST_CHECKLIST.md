# 最后一次SH测试：coord_scale=1 检查清单

**目标**: 测试coord_scale=1是否能改善SH+FBB性能

**假设**: 当前较大的coord_scale（可能10或15）导致SH密度计算时数值不稳定

---

## 第一步：确认当前配置

### 检查当前coord_scale

```bash
# 查找配置文件中的coord_scale设置
grep -r "coord_scale" configs/
grep -r "coord_scale" data/interpolant.py
```

**记录当前值**: ___________

### 检查SH密度计算代码

```bash
# 查看SH密度计算时是否使用了coord_scale
grep -A 10 "sh_density_from_atom14" data/interpolant.py
```

---

## 第二步：修改配置

### 修改coord_scale=1

**可能的位置**:
1. `configs/Train_SH.yaml` 或类似配置文件
2. `data/interpolant.py` 中的 `InterpolantConfig`

**修改内容**:
```yaml
# 在配置文件中
interpolant:
  coord_scale: 1.0  # 从之前的值改为1.0
```

或

```python
# 在interpolant.py中
self._cfg.coord_scale = 1.0
```

---

## 第三步：运行推理

### 使用现有checkpoint推理

**Checkpoint**:
```
/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__shdiffusion_decoder_ctx_shloss/2025-11-14_23-05-48/epoch=49-step=93900.ckpt
```

**命令**（示例）:
```bash
python inference.py \
  --config configs/Train_SH.yaml \
  --checkpoint /path/to/checkpoint \
  --output_dir outputs/shfbb_coordscale1_step10 \
  --num_steps 10
```

**重要**: 确认推理代码中SH密度计算使用了新的coord_scale

---

## 第四步：运行诊断

### 4.1 基础诊断指标

```bash
# 检查diagnostics.txt
ls outputs/shfbb_coordscale1_step10/val_seperated_*/sample_*/diagnostics.txt | head -3
```

**预期文件内容**:
- Sidechain RMSD
- Perplexity (pred vs GT)
- Recovery (pred vs GT)

### 4.2 运行对比分析

**创建对比脚本**（基于之前的代码）：

```bash
# 对比 coord_scale=1 vs 原始SH vs R3
python compare_coordscale_results.py
```

**对比内容**:
- coord_scale=1 的SH+FBB
- 原始 SH+FBB (outputs/shfbb_atoms_cords2_step10)
- R3 FBB (outputs/r3fbb_atoms_cords1_step10)

---

## 第五步：ESMFold评估（可选）

如果RMSD有改善，再跑ESMFold评估：

```bash
python evaluate_with_esmfold.py \
  --input_dir outputs/shfbb_coordscale1_step10/val_seperated_* \
  --output_dir outputs/shfbb_coordscale1_step10/esmfold_eval
```

然后对比：
```bash
python compare_sh_vs_r3_esmfold.py  # 修改路径包含新的coordscale1结果
```

---

## 关键对比指标

### 必须对比的指标

| 指标 | 原SH+FBB | coord_scale=1 | R3 FBB | 目标 |
|------|----------|---------------|---------|------|
| Sidechain RMSD | 2.31Å | ? | 1.06Å | <1.5Å |
| Perplexity | 4.73 | ? | 8.87 | - |
| Recovery | 64.3% | ? | 68.2% | >66% |
| pLDDT (ESMFold) | 52.45 | ? | 67.39 | >60 |

### 判断标准

**场景1: RMSD显著改善** (< 1.5Å)
- ✅ coord_scale是关键问题
- → 建议：使用coord_scale=1继续训练SH+FBB

**场景2: RMSD略有改善** (1.5-2.0Å)
- ⚠️  coord_scale有一定影响，但不是全部
- → 建议：测试其他coord_scale值（如5, 8）

**场景3: RMSD基本不变** (>2.0Å)
- ❌ coord_scale不是主要问题
- → 建议：放弃SH+FBB，专注R3 FBB

---

## 调试检查点

### 如果结果仍然差

**检查1**: SH密度是否真的使用了新的coord_scale？

```python
# 在data/interpolant.py的fbb_sample_iterative中添加print
print(f"Debug: coord_scale = {self._cfg.coord_scale}")
print(f"Debug: atoms14_local_t range = [{atoms14_local_t.min():.2f}, {atoms14_local_t.max():.2f}]")
```

**检查2**: 是否需要重新训练而非仅推理？

- coord_scale在训练时使用
- 如果模型在训练时用了大的coord_scale
- 推理时改小可能不兼容

**解决**: 可能需要用coord_scale=1重新训练几个epoch

---

## 备选测试

### 如果直接改coord_scale不行

**测试GT SH密度推理**:

修改 `data/interpolant.py` 的 `fbb_sample_iterative`:

```python
# 在循环前计算GT SH（一次性）
normalize_density_gt, *_ = sh_density_from_atom14_with_masks_clean(
    batch['atoms14_local'],  # 使用GT坐标
    batch['atom14_element_idx'],
    batch['atom14_gt_exists'],
    L_max=8, R_bins=24,
)
normalize_density_gt = normalize_density_gt / torch.sqrt(torch.tensor(4 * torch.pi))

# 在采样循环中
for i in range(len(ts) - 1):
    # ... 其他代码 ...

    # 使用固定的GT SH，而不是从噪声坐标计算
    input_feats['normalize_density'] = normalize_density_gt

    out = model(input_feats)
    # ... 其他代码 ...
```

**预期**: 如果RMSD接近1.0Å → 完全确认SH不稳定问题

---

## 时间估算

- 修改配置: 10分钟
- 运行推理 (10步): 30-60分钟
- 运行诊断分析: 10分钟
- **总计: ~1.5小时**

如果需要ESMFold评估: +1小时

---

## 输出文档

完成后创建:

```
analysis/SH_COORDSCALE1_RESULTS.md
  ├─ 配置变更
  ├─ 诊断结果对比表
  ├─ 结论和建议
  └─ 是否继续SH方向的决策
```

---

**准备好了就开始！**

记得：
1. 备份原配置
2. 记录所有修改
3. 保存所有输出
4. 对比关键指标

**Good luck!** 🚀
