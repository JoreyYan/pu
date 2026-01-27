import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_enclosing_ellipse(points):
    """
    计算包裹点集的椭圆参数
    返回: (中心x, 中心y), 宽, 高, 旋转角度(度)
    """
    points = np.array(points)

    # 1. 计算中心
    center = points.mean(axis=0)

    # 2. 计算协方差矩阵
    # rowvar=False 表示每一列是一个变量(x, y)，每一行是一个样本
    cov = np.cov(points, rowvar=False)

    # 3. 计算特征值和特征向量
    lambda_, v = np.linalg.eigh(cov)

    # 排序特征值，确保 lambda_[1] 是最大的（长轴）
    order = lambda_.argsort()
    lambda_ = lambda_[order]
    v = v[:, order]

    # 计算旋转角度 (长轴特征向量的角度)
    # v[:, 1] 是对应最大特征值的特征向量
    angle = np.degrees(np.arctan2(*v[:, 1][::-1]))

    # 4. 计算缩放系数以包裹所有点
    # 将所有点转换到椭圆的主轴坐标系中
    # 公式：T = (Points - Center) * Eigenvectors
    # 这样变换后，x轴对应短轴，y轴对应长轴（因为我们上面排序了）
    centered_points = points - center
    projected = np.dot(centered_points, v)

    # 计算每个点在归一化椭圆方程下的距离: x^2/lambda_0 + y^2/lambda_1
    # 我们需要找到一个系数 S，使得 max(x^2/(S*lambda_0) + y^2/(S*lambda_1)) = 1
    # 等价于求 radii_scale = max( x^2/lambda_0 + y^2/lambda_1 )

    radii_squares = (projected ** 2) / lambda_
    scale_factor = np.max(np.sum(radii_squares, axis=1))

    # 计算最终的长短轴长度
    # 宽度对应 lambda_[0] (短轴), 高度对应 lambda_[1] (长轴)
    # width/height 是直径，所以要 * 2
    width = 2 * np.sqrt(scale_factor * lambda_[0])
    height = 2 * np.sqrt(scale_factor * lambda_[1])

    return center, width, height, angle


# --- 测试代码 ---

# 1. 生成一些具有线性趋势的随机点
np.random.seed(42)
n_points = 50
# 生成 x 数据
x = np.random.normal(0, 5, n_points)
# 生成 y 数据，让它与 x 强相关 (线性排布) 并加一点噪声
y = 1.5 * x + np.random.normal(0, 2, n_points)
points = np.column_stack((x, y))

# 2. 计算椭圆
center, width, height, angle = get_enclosing_ellipse(points)

# 3. 绘图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(points[:, 0], points[:, 1], label='Data Points')

# 创建椭圆 Patch
ell = Ellipse(xy=center, width=width, height=height, angle=angle,
              edgecolor='red', facecolor='none', lw=2, label='Enclosing Ellipse')
ax.add_patch(ell)

# 标记中心和长轴方向
ax.plot(center[0], center[1], 'r+', markersize=10)

# 设置图形属性
ax.set_aspect('equal')
ax.legend()
ax.grid(True)
plt.title(f"Enclosing Ellipse (Angle: {angle:.2f}°)")
plt.show()