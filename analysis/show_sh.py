import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import math


def normed_vec(V: torch.Tensor, distance_eps: float = 1e-3) -> torch.Tensor:
    """Normalized vectors with distance smoothing."""
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U


def normed_cross(V1: torch.Tensor, V2: torch.Tensor, distance_eps: float = 1e-3) -> torch.Tensor:
    """Normalized cross product between vectors."""
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C


def frames_from_backbone(X: torch.Tensor, distance_eps: float = 1e-3):
    """Convert a backbone into local reference frames."""
    X_N, X_CA, X_C, X_O = X.unbind(-2)
    u_CA_N = normed_vec(X_N - X_CA, distance_eps)
    u_CA_C = normed_vec(X_C - X_CA, distance_eps)
    n_1 = u_CA_N
    n_2 = normed_cross(n_1, u_CA_C, distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps)
    R = torch.stack([n_1, n_2, n_3], -1)
    return R, X_CA


def cartesian_to_spherical(xyz):
    """Convert cartesian coordinates to spherical coordinates."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-8)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1, 1))  # polar angle [0, π]
    phi = torch.atan2(y, x)  # azimuthal angle [-π, π]
    return r, theta, phi


def compute_spherical_harmonics(theta, phi, L_max=2):
    """Compute spherical harmonics Y_l^m(theta, phi) for l=0 to L_max."""
    harmonics = {}

    # Convert to numpy for scipy computation
    theta_np = theta.detach().numpy()
    phi_np = phi.detach().numpy()

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # scipy uses (m, l, phi, theta) order, and phi/theta are swapped
            Y_lm = sph_harm(m, l, phi_np, theta_np)
            harmonics[(l, m)] = torch.from_numpy(Y_lm.real.astype(np.float32))

    return harmonics


def project_to_sh_density(atoms_local, atom_types, L_max=2, R_bins=16, r_max=10.0):
    """Project atoms to spherical harmonics density representation.

    Args:
        atoms_local: Local atom coordinates relative to CA frame, shape (N_atoms, 3)
        atom_types: Atom type indices (0=C, 1=N, 2=O, 3=S), shape (N_atoms,)
        L_max: Maximum spherical harmonics order
        R_bins: Number of radial bins
        r_max: Maximum radius to consider

    Returns:
        density_sh: Shape (C, L_max+1, 2*L_max+1, R_bins)
    """
    C = 4  # Number of atom type channels (C, N, O, S)

    # Initialize density tensor
    density_sh = torch.zeros(C, L_max + 1, 2 * L_max + 1, R_bins)

    # Convert to spherical coordinates
    r, theta, phi = cartesian_to_spherical(atoms_local)

    # Compute spherical harmonics
    harmonics = compute_spherical_harmonics(theta, phi, L_max)

    # Radial binning
    r_bins = torch.linspace(0, r_max, R_bins + 1)

    for i, (atom_r, atom_type) in enumerate(zip(r, atom_types)):
        if atom_r > r_max:
            continue

        # Find radial bin
        r_bin_idx = torch.clamp(
            torch.floor(atom_r / r_max * R_bins).long(),
            0, R_bins - 1
        )

        # Add contribution to all spherical harmonics
        l_idx = 0
        for l in range(L_max + 1):
            m_idx = 0
            for m in range(-l, l + 1):
                # Get spherical harmonic value for this atom
                Y_lm = harmonics[(l, m)][i]

                # Add Gaussian-like contribution in radial direction
                sigma_r = 0.5  # Radial width
                for r_idx in range(R_bins):
                    r_center = (r_idx + 0.5) * r_max / R_bins
                    gauss_weight = torch.exp(-0.5 * ((atom_r - r_center) / sigma_r) ** 2)

                    density_sh[atom_type, l_idx, m_idx, r_idx] += Y_lm * gauss_weight

                m_idx += 1
            l_idx += 1

    return density_sh


def visualize_density_peaks(density_sh, L_max=2, R_bins=16, r_max=10.0):
    """Visualize density peaks in 3D space by reconstructing from SH coefficients."""

    # Create a 3D grid for visualization
    n_points = 32
    x = torch.linspace(-r_max, r_max, n_points)
    y = torch.linspace(-r_max, r_max, n_points)
    z = torch.linspace(-r_max, r_max, n_points)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    # Convert grid to spherical coordinates
    r_grid, theta_grid, phi_grid = cartesian_to_spherical(grid_coords)

    # Compute spherical harmonics for grid
    harmonics_grid = compute_spherical_harmonics(theta_grid, phi_grid, L_max)

    # Reconstruct density at grid points
    density_grid = torch.zeros(len(grid_coords))

    for c in range(4):  # For each atom type channel
        l_idx = 0
        for l in range(L_max + 1):
            m_idx = 0
            for m in range(-l, l + 1):
                Y_lm = harmonics_grid[(l, m)]

                # Interpolate radial component
                for i, r_val in enumerate(r_grid):
                    if r_val < r_max:
                        r_bin_float = r_val / r_max * R_bins
                        r_bin_low = int(torch.floor(r_bin_float))
                        r_bin_high = min(r_bin_low + 1, R_bins - 1)

                        if r_bin_low < R_bins:
                            # Linear interpolation in radial direction
                            weight = r_bin_float - r_bin_low
                            radial_val = (1 - weight) * density_sh[c, l_idx, m_idx, r_bin_low] + \
                                         weight * density_sh[c, l_idx, m_idx, r_bin_high]

                            density_grid[i] += Y_lm[i] * radial_val

                m_idx += 1
            l_idx += 1

    # Reshape back to 3D grid
    density_3d = density_grid.reshape(n_points, n_points, n_points)

    return X, Y, Z, density_3d


# Demo: ASP residue from the PDB data
def demo_asp_residue():
    """Demo with the provided ASP residue."""

    # ASP residue atoms from PDB
    atoms_coords = torch.tensor([
        [15.568, 14.497, 27.361],  # N
        [14.432, 13.864, 28.096],  # CA
        [13.086, 14.425, 27.705],  # C
        [12.754, 15.541, 28.086],  # O
        [14.606, 14.061, 29.595],  # CB
        [15.592, 13.104, 30.190],  # CG
        [16.744, 13.040, 29.685],  # OD1
    ])

    # Atom types: 0=C, 1=N, 2=O, 3=S
    atom_types = torch.tensor([1, 0, 0, 2, 0, 0, 2])  # N, CA, C, O, CB, CG, OD1

    # Extract backbone atoms for frame calculation
    backbone = atoms_coords[:4].unsqueeze(0)  # (1, 4, 3)

    # Compute local frame
    R, CA = frames_from_backbone(backbone)
    R = R.squeeze(0)  # (3, 3)
    CA = CA.squeeze(0)  # (3,)

    print(f"CA position: {CA}")
    print(f"Local frame R:\n{R}")

    # Transform all atoms to local coordinate system
    atoms_local = (atoms_coords - CA) @ R.T

    print(f"\nAtoms in local coordinates:")
    for i, (atom, atype) in enumerate(zip(atoms_local, atom_types)):
        atom_name = ['C', 'N', 'O', 'S'][atype]
        print(f"{i}: {atom_name} at {atom}")

    # Project to spherical harmonics density
    density_sh = project_to_sh_density(atoms_local, atom_types, L_max=2, R_bins=16)

    print(f"\nDensity tensor shape: {density_sh.shape}")
    print(f"Total density: {density_sh.sum():.3f}")

    # Print non-zero components
    print(f"\nNon-zero density components:")
    for c in range(4):
        for l in range(3):
            for m in range(2 * l + 1):
                for r in range(16):
                    val = density_sh[c, l, m, r]
                    if abs(val) > 1e-3:
                        atom_name = ['C', 'N', 'O', 'S'][c]
                        m_actual = m - l  # Convert to actual m value
                        print(f"  {atom_name} channel, l={l}, m={m_actual}, r={r}: {val:.4f}")

    # Visualize density peaks
    print("\nVisualizing density reconstruction...")
    X, Y, Z, density_3d = visualize_density_peaks(density_sh)

    # Find peak positions
    threshold = density_3d.max() * 0.1
    peak_mask = density_3d > threshold
    peak_indices = torch.nonzero(peak_mask)

    print(f"Found {len(peak_indices)} density peaks above threshold {threshold:.4f}")

    # Create visualization
    fig = plt.figure(figsize=(15, 5))

    # Plot 1: Original atom positions
    ax1 = fig.add_subplot(131, projection='3d')
    colors = ['red', 'blue', 'green', 'yellow']
    atom_names = ['C', 'N', 'O', 'S']

    for i, (pos, atype) in enumerate(zip(atoms_local, atom_types)):
        ax1.scatter(pos[0], pos[1], pos[2],
                    c=colors[atype], s=100, alpha=0.8,
                    label=f"{atom_names[atype]}" if i == 0 or atype not in atom_types[:i] else "")

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('Original Atom Positions')
    ax1.legend()

    # Plot 2: Density reconstruction (slice)
    ax2 = fig.add_subplot(132)
    mid_slice = density_3d[:, :, density_3d.shape[2] // 2]
    im = ax2.imshow(mid_slice, origin='lower', extent=[-10, 10, -10, 10], cmap='viridis')
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_title('Density Field (Z=0 slice)')
    plt.colorbar(im, ax=ax2)

    # Plot 3: Density vs distance from CA
    ax3 = fig.add_subplot(133)
    r_values = torch.linspace(0, 10, 100)
    total_density_r = torch.zeros_like(r_values)

    for i, r_val in enumerate(r_values):
        r_bin = int(r_val / 10 * 16)
        if r_bin < 16:
            total_density_r[i] = density_sh[:, 0, 0, r_bin].sum()  # l=0, m=0 component

    ax3.plot(r_values, total_density_r)
    ax3.set_xlabel('Distance from CA (Å)')
    ax3.set_ylabel('Density (l=0, m=0)')
    ax3.set_title('Radial Density Profile')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    return density_sh, atoms_local


if __name__ == "__main__":
    density_sh, atoms_local = demo_asp_residue()