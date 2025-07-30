# LoRA-X Baseline修正计划

## 目标
将当前的简化实现修正为与原文LoRA-X完全一致的baseline算法

## 当前问题分析

### 1. 核心算法缺陷
- **缺少源模型子空间**：当前只使用目标模型子空间 `U_t, V_t`
- **缺少旋转变换**：原文需要 `ΔΣt←s = Ut⊤ŪsΔΣsṼs⊤Vt`
- **维度处理不符合原文**：使用自定义投影而非伪逆方法

### 2. 原文算法要求

#### 相同维度情况
```
ΔWt←s = UtUt⊤ΔWsVtVt⊤ = UtUt⊤ŪsΔΣsṼs⊤VtVt⊤ = UtΔΣt←sVt⊤
```
其中：`ΔΣt←s = Ut⊤ŪsΔΣsṼs⊤Vt`

#### 不同维度情况
使用伪逆最小化Frobenius范数：
```
P̂ = arg minP ‖PŪs − Ut‖F²
```

## 修正计划

### 第一步：修正核心算法 (_transfer_single_layer)

**当前实现**：
```python
def _transfer_single_layer(self, lora_weight, source_base, target_base):
    U_t, _, Vh_t = self.compute_svd_subspace(target_base)
    V_t = Vh_t.T
    projected_weight = torch.mm(torch.mm(U_t, U_t.T), 
                               torch.mm(lora_weight, torch.mm(V_t, V_t.T)))
```

**修正为原文算法**：
```python
def _transfer_single_layer(self, lora_weight, source_base, target_base):
    # 计算源模型和目标模型的子空间
    U_s, S_s, Vh_s = self.compute_svd_subspace(source_base)
    U_t, S_t, Vh_t = self.compute_svd_subspace(target_base)
    V_s, V_t = Vh_s.T, Vh_t.T
    
    # 重构源模型的ΔΣs
    delta_sigma_s = torch.diag(S_s)  # 假设LoRA权重对应对角变换
    
    # 计算旋转变换 ΔΣt←s = Ut⊤ŪsΔΣsṼs⊤Vt
    delta_sigma_transfer = U_t.T @ U_s @ delta_sigma_s @ V_s.T @ V_t
    
    # 应用完整变换 ΔWt←s = UtΔΣt←sVt⊤
    transferred_weight = U_t @ delta_sigma_transfer @ V_t.T
    
    return transferred_weight
```

### 第二步：实现维度不匹配处理

**添加伪逆方法**：
```python
def _handle_dimension_mismatch(self, U_s, U_t, V_s, V_t):
    """处理维度不匹配情况，使用伪逆方法"""
    
    # 对于左奇异矩阵
    if U_s.shape != U_t.shape:
        # P̂ = arg minP ‖PŪs − Ut‖F²
        # 解为：P = Ut @ (U_s.T @ U_t).pinverse() @ U_s.T
        cross_product = U_s.T @ U_t
        P = U_t @ torch.pinverse(cross_product) @ U_s.T
        U_s_aligned = P @ U_s
    else:
        U_s_aligned = U_s
    
    # 对于右奇异矩阵
    if V_s.shape != V_t.shape:
        cross_product = V_s.T @ V_t
        Q = V_t @ torch.pinverse(cross_product) @ V_s.T
        V_s_aligned = Q @ V_s
    else:
        V_s_aligned = V_s
    
    return U_s_aligned, V_s_aligned
```

### 第三步：修正LoRA权重到ΔΣs的映射

**问题**：需要正确理解LoRA权重如何对应到SVD空间的ΔΣs

**解决方案**：
1. LoRA权重 = lora_B @ lora_A
2. 对完整权重进行SVD得到ΔΣs
3. 或者直接将LoRA权重视为低秩扰动

### 第四步：完整的算法流程

```python
def _transfer_single_layer_corrected(self, lora_weight, source_base, target_base):
    """完全符合原文的LoRA-X算法"""
    
    # 1. 计算源模型和目标模型的SVD子空间
    U_s, S_s, Vh_s = self.compute_svd_subspace(source_base)
    U_t, S_t, Vh_t = self.compute_svd_subspace(target_base)
    V_s, V_t = Vh_s.T, Vh_t.T
    
    # 2. 处理维度不匹配
    if U_s.shape != U_t.shape or V_s.shape != V_t.shape:
        U_s_aligned, V_s_aligned = self._handle_dimension_mismatch(U_s, U_t, V_s, V_t)
    else:
        U_s_aligned, V_s_aligned = U_s, V_s
    
    # 3. 将LoRA权重映射到SVD空间
    # 方法1：直接使用LoRA权重作为ΔWs
    delta_W_s = lora_weight
    
    # 方法2：或者先投影到源模型子空间
    # delta_W_s = U_s @ U_s.T @ lora_weight @ V_s @ V_s.T
    
    # 4. 计算旋转变换
    # ΔΣt←s = Ut⊤ŪsΔΣsṼs⊤Vt
    # 这里需要将ΔWs转换为ΔΣs
    delta_sigma_s = U_s_aligned.T @ delta_W_s @ V_s_aligned
    delta_sigma_transfer = U_t.T @ U_s_aligned @ delta_sigma_s @ V_s_aligned.T @ V_t
    
    # 5. 重构目标权重
    transferred_weight = U_t @ delta_sigma_transfer @ V_t.T
    
    return transferred_weight
```

## 实施步骤

1. **备份当前实现**：保存现有的简化版本
2. **修正核心算法**：按照原文公式重写 `_transfer_single_layer`
3. **添加维度处理**：实现伪逆方法处理维度不匹配
4. **测试验证**：确保算法与原文一致
5. **性能对比**：比较修正前后的迁移效果

## 预期结果

修正后的算法应该：
- 完全符合原文LoRA-X的数学公式
- 正确处理维度不匹配情况
- 提供更准确的子空间对齐
- 作为可靠的baseline用于后续改进

## 注意事项

1. **LoRA权重理解**：需要正确理解lora_A和lora_B如何组合成完整的ΔW
2. **数值稳定性**：伪逆计算可能存在数值不稳定问题
3. **计算效率**：原文算法比简化版本计算量更大
4. **内存使用**：双向子空间计算需要更多内存

## 验证标准

修正后的实现应该满足：
- [ ] 使用源模型和目标模型的双向子空间
- [ ] 实现旋转变换 ΔΣt←s
- [ ] 使用伪逆方法处理维度不匹配
- [ ] 数学公式与原文完全一致
- [ ] 通过单元测试验证