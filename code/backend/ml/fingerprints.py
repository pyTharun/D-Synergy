# ============================================================
# FINGERPRINTS MODULE
# Converts SMILES strings into Morgan molecular fingerprints
# using the modern RDKit API (no deprecation warnings)
# ============================================================

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# 1. CREATE A MORGAN FINGERPRINT GENERATOR
# ============================================================

def get_morgan_generator(radius: int = 2, fp_size: int = 2048):
    """
    Create and return a Morgan fingerprint generator.

    Args:
        radius  : Morgan radius (default=2 → captures 2 bonds away)
        fp_size : Number of bits in the fingerprint (default=2048)

    Returns:
        RDKit Morgan generator object
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size
    )
    return generator


# ============================================================
# 2. CONVERT A SINGLE SMILES → MORGAN FINGERPRINT ARRAY
# ============================================================

def generate_morgan_fingerprint(smiles: str, generator, fp_size: int = 2048) -> np.ndarray:
    """
    Convert a SMILES string into a fixed-length Morgan fingerprint array.

    Args:
        smiles    : Drug SMILES string
        generator : Morgan generator (from get_morgan_generator)
        fp_size   : Number of bits (must match generator config)

    Returns:
        numpy array of shape (fp_size,) — all zeros if SMILES is invalid
    """
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return np.zeros(fp_size, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None:
        # Invalid SMILES → return zero vector
        return np.zeros(fp_size, dtype=np.float32)

    fp = generator.GetFingerprint(mol)

    # Convert RDKit bit vector to numpy array
    arr = np.zeros(fp_size, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0

    return arr


# ============================================================
# 3. GENERATE FINGERPRINTS FOR A BATCH OF SMILES
# ============================================================

def generate_fingerprint_matrix(smiles_list, generator, fp_size: int = 2048) -> np.ndarray:
    """
    Generate Morgan fingerprints for a list/Series of SMILES strings.

    Args:
        smiles_list : iterable of SMILES strings
        generator   : Morgan generator
        fp_size     : fingerprint size

    Returns:
        numpy array of shape (n_samples, fp_size)
    """
    fps = [
        generate_morgan_fingerprint(smiles, generator, fp_size)
        for smiles in smiles_list
    ]
    matrix = np.vstack(fps).astype(np.float32)
    print(f"[Fingerprints] Generated matrix: {matrix.shape}")
    return matrix
