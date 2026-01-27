import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1) synthetic data: 2 clusters, 10 pts each
# ---------------------------
def make_two_clusters(n_per=10, seed=0):
    rng = np.random.default_rng(seed)

    # cluster 0: elongated along direction d0
    d0 = np.array([1.0, 0.3, 0.1])
    d0 = d0 / np.linalg.norm(d0)
    center0 = np.array([0.0, 0.0, 0.0])

    t0 = rng.normal(0, 4.0, size=(n_per, 1))
    noise0 = rng.normal(0, 0.6, size=(n_per, 3))
    pts0 = center0 + t0 * d0[None, :] + noise0

    # cluster 1: elongated along direction d1 (different)
    d1 = np.array([0.2, 1.0, -0.5])
    d1 = d1 / np.linalg.norm(d1)
    center1 = np.array([12.0, 8.0, -5.0])

    t1 = rng.normal(0, 5.0, size=(n_per, 1))
    noise1 = rng.normal(0, 0.7, size=(n_per, 3))
    pts1 = center1 + t1 * d1[None, :] + noise1

    pts = np.concatenate([pts0, pts1], axis=0)
    labels = np.array([0] * n_per + [1] * n_per, dtype=int)
    return pts, labels


# ---------------------------
# 2) PCA enclosing ellipsoid (statistical "exact-fit" by max Mahalanobis)
#    Returns center mu, radii (a,b,c), rotation matrix R (columns are axes in world)
# ---------------------------
def pca_enclosing_ellipsoid(points, eps=1e-9):
    points = np.asarray(points, dtype=np.float64)
    mu = points.mean(axis=0)

    X = points - mu
    cov = np.cov(X, rowvar=False)
    cov = 0.5 * (cov + cov.T)

    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, eps)

    order = np.argsort(evals)   # ascending: small->mid->large
    evals = evals[order]
    R = evecs[:, order]         # columns are world axes for each principal dir

    # ensure right-handed
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0

    # local coords
    local = X @ R

    md2 = np.sum((local**2) / evals[None, :], axis=1)
    scale = np.sqrt(np.max(md2))

    radii = np.sqrt(evals) * scale
    return mu, radii, R, cov


# ---------------------------
# 3) plot ellipsoid wireframe given (mu, radii, R)
# ---------------------------
def plot_ellipsoid(ax, mu, radii, R, color=None, alpha=0.25, wire=True):
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 24)
    xu = np.outer(np.cos(u), np.sin(v))
    yu = np.outer(np.sin(u), np.sin(v))
    zu = np.outer(np.ones_like(u), np.cos(v))

    x_loc = xu * radii[0]
    y_loc = yu * radii[1]
    z_loc = zu * radii[2]

    coords = np.stack([x_loc.ravel(), y_loc.ravel(), z_loc.ravel()], axis=0)  # [3,M]
    world = (R @ coords).T + mu[None, :]  # [M,3]

    Xw = world[:, 0].reshape(xu.shape)
    Yw = world[:, 1].reshape(xu.shape)
    Zw = world[:, 2].reshape(xu.shape)

    if wire:
        ax.plot_wireframe(Xw, Yw, Zw, rstride=2, cstride=2, color=color, alpha=alpha, linewidth=0.8)
    else:
        ax.plot_surface(Xw, Yw, Zw, color=color, alpha=alpha, linewidth=0)

    # draw long axis
    long_dir = R[:, 2]
    ax.quiver(mu[0], mu[1], mu[2],
              long_dir[0]*radii[2]*1.2, long_dir[1]*radii[2]*1.2, long_dir[2]*radii[2]*1.2,
              color=color, linewidth=2)


# ---------------------------
# 4) write points as atoms (no ANISOU)
# ---------------------------
def write_points_as_atoms(f, points, chain="A", start_serial=1000, resname="PTS"):
    """
    Write point cloud as HETATM records (no ANISOU).
    Returns next serial.
    """
    serial = start_serial
    for i, p in enumerate(points, start=1):
        x, y, z = p
        f.write(
            f"HETATM{serial:5d}  C   {resname:>3s} {chain}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
        )
        serial += 1
    return serial


# ---------------------------
# 5) write PDB + ANISOU for cluster centers, plus write all points
#    ANISOU uses Sigma = R diag(radii^2) R^T  (boundary-matching)
# ---------------------------



def _pdb_atom_name_4col(name: str, element: str = "C") -> str:
    """
    PDB 原子名称字段位于 13-16 列 (4 chars)。
    对于单个字母的元素 (C, N, O, S)，原子名称通常右对齐，前面留空。
    例如 'CA' 应该是 ' CA '。
    """
    name = name.strip()
    if len(name) == 2 and len(element) == 1:
        return f" {name:<2s} "  # ' CA '
    return f"{name:>4s}"


def write_pdb_with_anisou_and_points(
        filename,
        centers,
        covs,
        clusters_points,
        resnames=None,
):
    """
    按照 save_gaussian_as_pdb_strict 的严格格式风格，
    保存带有 ANISOU (椭球) 和 采样点 (HETATM) 的 PDB 文件。
    """

    # --- 1. 数据预处理 ---
    # 确保输入是 numpy 格式，避免后续 tensor/list 混用问题
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(covs, np.ndarray):
        covs = np.array(covs)

    N = len(centers)
    if resnames is None:
        resnames = ["GLY"] * N

    # 预处理原子名称格式 (Centroids use CA, Points use PT)
    aname_ca = _pdb_atom_name_4col("CA", element="C")
    aname_pt = _pdb_atom_name_4col("PT", element="H")  # 假设点云是 H 或者 dummy atom

    with open(filename, "w") as f:
        # --- 2. 写入 Header ---
        f.write("HEADER    PCA_ELLIPSOIDS_WITH_POINTS\n")
        # 写入虚拟 CRYST1，帮助部分可视化软件 (如 PyMOL) 保持正交视角
        f.write("CRYST1   90.000   90.000   90.000  90.00  90.00  90.00 P 1           1\n")

        serial = 1

        # --- 3. 写入中心点 (Centers) 与 协方差 (ANISOU) ---
        for i, (mu, C_mat) in enumerate(zip(centers, covs)):
            x, y, z = mu
            res = resnames[i]

            # 保持原逻辑：每个 Center 作为一个单独的 Chain (或者按需修改)
            # 注意：如果 i > 25，chr(ord('A') + i) 会变成非字母字符，PDB 规范中 Chain ID 只有 1 字符。
            # 这里做一个循环映射 A-Z, A-Z...
            chain_idx = i % 26
            chain = chr(ord("A") + chain_idx)

            # 残基编号
            resi = i + 1

            # 3.1 写入 ATOM 记录
            # Columns: 1-6 "ATOM  ", 7-11 serial, 13-16 name, 18-20 resName, 22 chain, 23-26 resSeq
            f.write(
                f"ATOM  {serial:5d} {aname_ca}{res:>3s} {chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{1.00:6.2f}           C  \n"
            )

            # 3.2 计算并写入 ANISOU 记录
            # PDB ANISOU 单位是 1e-4 Angstrom^2
            U = C_mat  # 假设输入已经是 Covariance Matrix

            # 防御性对称化
            U = 0.5 * (U + U.T)

            # 转换为整数，使用 np.round 避免 int() 直接截断带来的误差
            u_vals = U * 1e4
            u11 = int(np.round(u_vals[0, 0]))
            u22 = int(np.round(u_vals[1, 1]))
            u33 = int(np.round(u_vals[2, 2]))
            u12 = int(np.round(u_vals[0, 1]))
            u13 = int(np.round(u_vals[0, 2]))
            u23 = int(np.round(u_vals[1, 2]))

            # 溢出检查 (Strict Check)
            arr = np.array([u11, u22, u33, u12, u13, u23], dtype=np.int64)
            if np.any(arr > 9_999_999) or np.any(arr < -9_999_999):
                print(
                    f"[WARN] ANISOU overflow risk for atom {serial}. "
                    f"Max val: {np.max(np.abs(arr))}. Ellipsoid may look wrong."
                )

            # 格式化 ANISOU (7字符宽度)
            f.write(
                f"ANISOU{serial:5d} {aname_ca}{res:>3s} {chain}{resi:4d} "
                f"{u11:7d}{u22:7d}{u33:7d}{u12:7d}{u13:7d}{u23:7d}       C  \n"
            )

            serial += 1

        # --- 4. 写入采样点 (Points as HETATM) ---
        # 对应每个 Cluster 的点
        if clusters_points is not None:
            for k, pts in enumerate(clusters_points):
                # 保持与 Center 相同的 Chain ID
                chain_idx = k % 26
                chain = chr(ord("A") + chain_idx)

                # 残基编号继续或者使用特定的编号，这里使用 Cluster ID + 1
                resi = k + 1

                # 确保 pts 是 numpy array
                if not isinstance(pts, np.ndarray):
                    # 如果是 tensor 转 numpy
                    if hasattr(pts, 'detach'):
                        pts = pts.detach().cpu().numpy()
                    else:
                        pts = np.array(pts)

                if len(pts) == 0:
                    continue

                for p_idx, point in enumerate(pts):
                    px, py, pz = point

                    # 写入 HETATM
                    # ResName 使用 "PTS" 区分
                    f.write(
                        f"HETATM{serial:5d} {aname_pt}PTS {chain}{resi:4d}    "
                        f"{px:8.3f}{py:8.3f}{pz:8.3f}"
                        f"{1.00:6.2f}{1.00:6.2f}           H  \n"
                    )
                    serial += 1

        f.write("END\n")

    print(f"Saved PDB with ANISOU and Points to: {filename}")


# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    pts, labels = make_two_clusters(n_per=10, seed=42)

    clusters = [0, 1]
    colors = ["tab:blue", "tab:orange"]

    centers = []
    covs_for_pdb = []

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for k, color in zip(clusters, colors):
        P = pts[labels == k]
        mu, radii, R, cov = pca_enclosing_ellipsoid(P)

        # For PDB: use Sigma = R diag(radii^2) R^T (matches drawn boundary)
        Sigma = R @ np.diag(radii**2) @ R.T
        Sigma = 0.5 * (Sigma + Sigma.T)

        centers.append(mu)
        covs_for_pdb.append(Sigma)

        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, color=color, alpha=0.7)
        ax.scatter(mu[0], mu[1], mu[2], s=120, color="k", marker="x")
        plot_ellipsoid(ax, mu, radii, R, color=color, alpha=0.35, wire=True)

        print(f"\nCluster {k}:")
        print("  mu:", mu)
        print("  radii (small,mid,large):", radii)
        print("  long axis dir:", R[:, 2])
        print("  Sigma:\n", Sigma)

    # set roughly equal aspect
    allc = pts
    max_range = (allc.max(0) - allc.min(0)).max() / 2.0
    mid = (allc.max(0) + allc.min(0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    plt.title("PCA Enclosing Ellipsoids (plt)")

    # write PDB with 2 atoms (centers) + all random points
    clusters_points = [pts[labels == 0], pts[labels == 1]]
    pdb_path = "two_clusters_ellipsoids_with_points.pdb"
    write_pdb_with_anisou_and_points(
        pdb_path,
        centers,
        covs_for_pdb,
        clusters_points,
        resnames=["ALA", "VAL"],
    )
    print("\nWrote PDB:", pdb_path)
    print("\nIn PyMOL:")
    print(f"  load {pdb_path}")
    print("  hide everything")
    print("  show spheres")
    print("  show ellipsoids")
    print("  set ellipsoid_scale, 1.0")
    print("  set sphere_scale, 0.3")
    print("  color blue, chain A")
    print("  color orange, chain B")

    # NOTE: savefig should be before show() in most backends
    plt.savefig("two_clusters_ellipsoids.png", dpi=200)
    plt.show()
