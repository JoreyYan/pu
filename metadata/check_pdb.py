import pandas as pd
import os
from pathlib import Path


def check_files_exist(csv_file_path, metadir_path):
    """
    检查CSV文件中列出的pkl文件是否在指定目录下存在

    Args:
        csv_file_path: CSV文件的路径
        metadir_path: metadir目录的路径

    Returns:
        dict: 包含检查结果的字典
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取CSV文件，共 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None

    # 检查是否有processed_path列
    if 'processed_path' not in df.columns:
        print("CSV文件中没有找到 'processed_path' 列")
        return None

    # 获取所有文件路径
    file_paths = df['processed_path'].tolist()

    # 初始化结果统计
    results = {
        'total_files': len(file_paths),
        'existing_files': 0,
        'missing_files': 0,
        'existing_list': [],
        'missing_list': []
    }

    print(f"\n开始检查 {results['total_files']} 个文件...")
    print(f"基础目录: {metadir_path}")
    print("-" * 50)

    # 逐个检查文件是否存在
    for i, file_path in enumerate(file_paths, 1):
        # 构建完整的文件路径
        full_path = os.path.join(metadir_path, file_path)

        # 检查文件是否存在
        if os.path.exists(full_path):
            results['existing_files'] += 1
            results['existing_list'].append(file_path)
            status = "✓ 存在"
        else:
            results['missing_files'] += 1
            results['missing_list'].append(file_path)
            status = "✗ 缺失"

        print(f"{i:3d}. {file_path} - {status}")

    # 打印总结
    print("\n" + "=" * 60)
    print("检查结果总结:")
    print(f"总文件数: {results['total_files']}")
    print(f"存在文件数: {results['existing_files']}")
    print(f"缺失文件数: {results['missing_files']}")
    print(f"完整性: {results['existing_files'] / results['total_files'] * 100:.1f}%")

    # 如果有缺失文件，详细列出
    if results['missing_files'] > 0:
        print(f"\n缺失的文件列表:")
        for missing_file in results['missing_list']:
            print(f"  - {missing_file}")

    return results


def main():
    # 设置文件路径 - 请根据实际情况修改这些路径
    csv_file_path = '/home/junyu/project/protein-frame-flow-u/metadata/pdb_metadata.csv'  # 替换为你的CSV文件路径
    metadir_path = '/media/junyu/DATA/frameflow2/'  # 替换为你的metadir目录路径

    # 检查输入路径是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: CSV文件不存在 - {csv_file_path}")
        return

    if not os.path.exists(metadir_path):
        print(f"错误: metadir目录不存在 - {metadir_path}")
        return

    # 执行文件检查
    results = check_files_exist(csv_file_path, metadir_path)

    if results:
        # 可选：将结果保存到文件
        save_results = input("\n是否要将缺失文件列表保存到文件? (y/n): ").lower().strip()
        if save_results == 'y' and results['missing_files'] > 0:
            missing_file_path = "missing_files.txt"
            with open(missing_file_path, 'w', encoding='utf-8') as f:
                f.write("缺失的文件列表:\n")
                f.write("=" * 40 + "\n")
                for missing_file in results['missing_list']:
                    f.write(f"{missing_file}\n")
            print(f"缺失文件列表已保存到: {missing_file_path}")


if __name__ == "__main__":
    main()