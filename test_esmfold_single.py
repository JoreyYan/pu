"""
测试ESMFold能否正常工作
"""
import os
import sys

sys.path.insert(0, '/home/junyu/project/esm')
sys.path.insert(0, '/home/junyu/project/esm/genie/evaluations/pipeline')

from fold_models.esmfold import ESMFold

# 测试序列（从SDE结果中提取的）
test_seq = "TRPPFHIVIPLYPGVNDDNVAAPVKIFSWLADAAEAFAVTITLAAEHNTPIETRDGNTLTPQREFADFADAAAPQPKVHIIWVPGGAPDVIRKNMRGGPYLDFIKAESAGADFVSSTSKGALILAAAGLLDGYRATTDWAFLPSLQQFPAIQVAEGRPFYTLDGDFIVGGGISSGLAEALALVARTAGENIAEHVKLLTEFFPTITPATHSPLK"

print("初始化ESMFold...")
model = ESMFold()

print(f"测试序列长度: {len(test_seq)}")
print("开始折叠...")

try:
    pdb_str, pae = model.predict(test_seq)
    print(f"折叠成功!")
    print(f"PDB字符串长度: {len(pdb_str)}")
    print(f"PAE shape: {pae.shape if pae is not None else None}")

    # 保存测试结果
    output_path = "/home/junyu/project/pu/test_esmfold_output.pdb"
    with open(output_path, 'w') as f:
        f.write(pdb_str)
    print(f"已保存到: {output_path}")

except Exception as e:
    print(f"折叠失败: {e}")
    import traceback
    traceback.print_exc()
