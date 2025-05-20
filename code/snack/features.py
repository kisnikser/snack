import torch
from functools import lru_cache


# Признаковое пространство для аминокислот
class FeatureSpace:
    def __init__(self, device=None):
        self.feature_dim = 5
        self.device = device  # Device can be specified during initialization or auto-detected later
        self._feature_cache = (
            {}
        )  # Cache for feature vectors to avoid redundant computation

        # Standard amino acids list
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY-"

        # Гидрофобность по шкале Kyte-Doolittle
        self.hydropathy_index = {
            "A": 1.8,
            "C": 2.5,
            "D": -3.5,
            "E": -3.5,
            "F": 4.5,
            "G": -0.4,
            "H": -0.5,
            "I": 4.5,
            "K": -3.9,
            "L": 4.2,
            "M": 1.9,
            "N": -3.5,
            "P": -1.6,
            "Q": -3.5,
            "R": -4.5,
            "S": -0.8,
            "T": -0.7,
            "V": 4.2,
            "W": -0.9,
            "Y": -1.3,
            "-": 0.0,  # Gap character
        }

        # Молекулярная масса
        self.molecular_weight = {
            "A": 89.09,
            "C": 121.16,
            "D": 133.10,
            "E": 147.13,
            "F": 165.19,
            "G": 75.07,
            "H": 155.15,
            "I": 131.17,
            "K": 146.19,
            "L": 131.17,
            "M": 149.21,
            "N": 132.12,
            "P": 115.13,
            "Q": 146.15,
            "R": 174.19,
            "S": 105.09,
            "T": 119.12,
            "V": 117.15,
            "W": 204.23,
            "Y": 181.19,
            "-": 0.0,  # Gap character
        }

        # Полярность аминокислот
        self.polarity = {
            "A": 0,
            "C": 0,
            "D": 1,
            "E": 1,
            "F": -1,
            "G": 0,
            "H": 1,
            "I": -1,
            "K": 1,
            "L": -1,
            "M": -1,
            "N": 1,
            "P": 0,
            "Q": 1,
            "R": 1,
            "S": 0,
            "T": 0,
            "V": -1,
            "W": -1,
            "Y": 0,
            "-": 0,  # Gap character
        }

        # Заряд при pH=7
        self.charge_at_pH_7 = {
            "A": 0,
            "C": 0,
            "D": -1,
            "E": -1,
            "F": 0,
            "G": 0,
            "H": 1,
            "I": 0,
            "K": 1,
            "L": 0,
            "M": 0,
            "N": 0,
            "P": 0,
            "Q": 0,
            "R": 1,
            "S": 0,
            "T": 0,
            "V": 0,
            "W": 0,
            "Y": 0,
            "-": 0,  # Gap character
        }

        # Объем аминокислот
        self.volume = {
            "A": 67,
            "C": 108,
            "D": 111,
            "E": 138,
            "F": 165,
            "G": 48,
            "H": 155,
            "I": 140,
            "K": 154,
            "L": 140,
            "M": 162,
            "N": 114,
            "P": 112,
            "Q": 146,
            "R": 168,
            "S": 105,
            "T": 119,
            "V": 141,
            "W": 220,
            "Y": 193,
            "-": 0,  # Gap character
        }

        # Pre-compute all features for known amino acids for faster access
        self._precompute_features()

    def _precompute_features(self):
        """Pre-compute feature vectors for all standard amino acids"""
        for aa in self.amino_acids:
            self._compute_features(aa)

    def to(self, device):
        """Move the feature vectors to the specified device"""
        self.device = device
        # Clear and rebuild the cache on the new device
        self._feature_cache = {}
        self._precompute_features()
        return self

    @lru_cache(maxsize=256)
    def _compute_features(self, amino_acid):
        """Compute the feature vector for an amino acid (without caching)"""
        hydropathy = self.hydropathy_index.get(amino_acid, 0.0)
        molecular_weight = self.molecular_weight.get(amino_acid, 0.0)
        polarity = self.polarity.get(amino_acid, 0.0)
        charge = self.charge_at_pH_7.get(amino_acid, 0.0)
        volume = self.volume.get(amino_acid, 0.0)

        # Standardize features
        molecular_weight = (
            molecular_weight / 200.0
        )  # Normalize to approximately [0, 1] range
        volume = volume / 220.0  # Normalize to approximately [0, 1] range

        # Create the feature vector
        features = torch.tensor(
            [hydropathy, molecular_weight, polarity, charge, volume],
            dtype=torch.float,
        )

        # Move to the appropriate device if specified
        if self.device is not None:
            features = features.to(self.device)

        return features

    def get_features(self, amino_acid):
        """Return the feature vector for an amino acid, with caching for performance"""
        # If amino_acid is already a tensor, return it
        if isinstance(amino_acid, torch.Tensor):
            return amino_acid

        # Handle integer indices by converting to amino acid characters
        if isinstance(amino_acid, int) and 0 <= amino_acid < len(self.amino_acids):
            amino_acid = self.amino_acids[amino_acid]

        # Ensure amino_acid is a string at this point
        if not isinstance(amino_acid, str):
            # Create a default zero vector for unknown inputs
            zero_vec = torch.zeros(self.feature_dim, dtype=torch.float)
            return zero_vec.to(self.device) if self.device else zero_vec

        # Check if we've already computed this feature vector
        if amino_acid not in self._feature_cache:
            self._feature_cache[amino_acid] = self._compute_features(amino_acid)

        return self._feature_cache[amino_acid]
