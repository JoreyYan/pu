# 快速参考：诊断结果摘要

## 核心发现（3句话总结）

1. **框架正确**：ODE积分完全正确（RMSE<0.001Å），不需要迁移
2. **SH是瓶颈**：SH+FBB性能差R3 FBB 2倍（2.31Å vs 1.06Å）
3. **Velocity方向误差**：导致增加步数无法改善（1.2-2.4Å方向误差）

---

## 关键数据对比

### SH+FBB vs R3 FBB (10步)

| 指标 | SH+FBB | R3 FBB | 差异 |
|------|--------|--------|------|
| Sidechain RMSD | 2.31Å | 1.06Å | **R3好2.2倍** |
| pLDDT (ESMFold) | 52.45 | 67.39 | **R3高28%** |
| TM-score | 0.453 | 0.619 | **R3高37%** |
| Recovery | 64.3% | 68.2% | R3高4% |

### R3 FBB步数对比

| 指标 | 10步 | 100步 | 500步 |
|------|------|-------|-------|
| RMSD | 1.059Å | 1.093Å | 1.077Å |
| C-N键误差 | 0.165Å | 0.137Å | 0.139Å |
| 侧链clash | 244次 | 241次 | **373次** |

**结论**: 增加步数对RMSD无改善，500步clash恶化53%

---

## 代码清单

### 运行诊断（按顺序）

```bash
# 1. 框架和velocity验证
python quick_framework.py

# 2. R3多步数对比
python compare_r3_steps.py

# 3. Sequence-结构一致性
python check_sequence_structure_consistency.py

# 4. 侧链几何质量
python check_sidechain_geometry.py

# 5. ESMFold评估（R3多步数）
python compare_esmfold_results.py

# 6. ESMFold评估（SH vs R3）
python compare_sh_vs_r3_esmfold.py
```

### 主要输出目录

```
R3 FBB:
  outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156/
  outputs/r3fbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251116_210400/
  outputs/r3fbb_atoms_cords1_step500/val_seperated_Rm0_t0_step0_20251116_211051/

SH+FBB:
  outputs/shfbb_atoms_cords2_step10/val_seperated_Rm0_t0_step0_20251116_185102/
  outputs/shfbb_atoms_cords2_step100/val_seperated_Rm0_t0_step0_20251116_185403/
```

---

## 下一步行动

### 最后一次SH尝试
**参数**: `coord_scale=1`（当前可能是10或15）

**预期**:
- 如果RMSD→1.0Å：coord_scale是问题
- 如果RMSD仍2.3Å：SH密度本身限制

### 如果SH仍差
**建议**:
1. ✅ 专注R3 FBB（已证明更优）
2. 改进velocity训练（降低1.2-2.4Å方向误差）
3. 添加几何约束loss（改善C-N/C-O键长）

---

## 诊断证据链

```
实验1: 框架正确性
  ├─ 真实velocity + ODE → RMSE<0.001Å
  └─ ✅ 框架无问题

实验2: Velocity质量
  ├─ Norm: 3-4Å ✓
  ├─ 方向: 1.2-2.4Å误差 ❌
  └─ → 这是增加步数无效的原因

实验4+6: R3多步数
  ├─ RMSD: 1.06→1.09Å（无改善）
  ├─ C-N键: 改善17%
  ├─ Clash: 500步恶化53%
  └─ → Velocity方向误差累积

实验8+9: SH vs R3
  ├─ RMSD: 2.31 vs 1.06Å（差2.2倍）
  ├─ pLDDT: 52 vs 67（差28%）
  └─ ✅ SH密度是主要瓶颈
```

---

完整报告见: `analysis/DIAGNOSTIC_REPORT.md`
