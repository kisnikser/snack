import torch
import torch.nn as nn
from functools import lru_cache
from .alignment import needleman_wunsch


# Метрическая функция расстояния
class Snack(nn.Module):  # Наследуемся от torch.nn.Module
    def __init__(self, feature_space, lambda1=1.0, lambda2=1.0):
        super().__init__()  # Инициализируем родительский класс
        self.feature_space = feature_space
        self.lambda1 = lambda1  # Вес для асимметрии
        self.lambda2 = lambda2  # Вес для нарушения неравенства треугольника

        # Инициализация матрицы M как положительно полуопределенной
        self.M = nn.Parameter(
            torch.eye(self.feature_space.feature_dim)
        )  # Инициализация M как единичной матрицы

        # Cache for feature vectors to avoid redundant computation
        self._features_cache = {}

    def to(self, device):
        """Move the model to the specified device and clear caches"""
        # Clear feature cache when moving to a new device
        self._features_cache = {}
        return super().to(device)

    @lru_cache(maxsize=256)
    def get_features(self, char):
        """Get feature vector for a character, with caching"""
        # Convert numeric indices to amino acid characters if needed
        if isinstance(char, int) and 0 <= char < 20:
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            char = amino_acids[char]

        # Move feature to the same device as model parameters
        device = next(self.parameters()).device
        features = self.feature_space.get_features(char)
        return features.to(device)

    def __call__(self, i, j):
        """Вычисляем расстояние между аминокислотами a_i и a_j"""
        # Get feature vectors for the input characters
        try:
            x_i = self.get_features(i)
            x_j = self.get_features(j)

            # Ensure both are on the same device as the model
            device = self.M.device
            x_i = x_i.to(device)
            x_j = x_j.to(device)
        except Exception as e:
            # Fallback for handling problematic inputs
            if isinstance(i, (int, float)) and isinstance(j, (int, float)):
                # For numeric pairs that aren't amino acid indices, return a default distance
                return torch.tensor(abs(i - j), device=self.M.device)
            else:
                raise e

        delta_ij = x_i - x_j
        distance = torch.matmul(delta_ij, torch.matmul(self.M, delta_ij))
        normalized_distance = torch.tanh(distance)
        return normalized_distance

    def alignment_loss(self, alignment_data):
        """Функция ошибки для выравнивания"""
        total_loss = 0
        for s1, s2 in alignment_data:
            # Используем Needleman-Wunsch для получения выравнивания
            aligned_seq1, aligned_seq2 = needleman_wunsch(s1, s2, self)
            loss = self.alignment_loss_function(aligned_seq1, aligned_seq2)
            total_loss += loss
        return total_loss

    def alignment_loss_function(self, aligned_seq1, aligned_seq2):
        """Функция для вычисления ошибки выравнивания"""
        loss = 0
        # Проходим по всем позициям выравнивания и сравниваем
        for a, b in zip(aligned_seq1, aligned_seq2):
            if a == b:
                loss += 0  # Нет ошибки, если совпадают
            else:
                loss += self(a, b)  # Потери по метрике для несовпадений

                # Штраф за инсерты или делиты
                if a == "-" or b == "-":
                    loss += self.lambda1  # Штраф за пустое место
        return loss

    def asymmetry_loss(self):
        """Функция ошибки для асимметрии"""
        asymmetry_loss = 0
        # Use actual amino acid characters instead of indices for more robust calculation
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        for i in range(len(amino_acids)):
            for j in range(i + 1, len(amino_acids)):
                aa_i = amino_acids[i]
                aa_j = amino_acids[j]
                loss = torch.abs(self(aa_i, aa_j) - self(aa_j, aa_i))
                asymmetry_loss += loss
        return asymmetry_loss

    def triangle_loss(self):
        """Функция ошибки для нарушения неравенства треугольника"""
        triangle_loss = 0
        # Use actual amino acid characters instead of indices for more robust calculation
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        device = self.M.device

        for i in range(len(amino_acids)):
            for j in range(i + 1, len(amino_acids)):
                for k in range(j + 1, len(amino_acids)):
                    aa_i = amino_acids[i]
                    aa_j = amino_acids[j]
                    aa_k = amino_acids[k]

                    dist_ij = self(aa_i, aa_j)
                    dist_jk = self(aa_j, aa_k)
                    dist_ik = self(aa_i, aa_k)

                    # Ensure tensor is on the right device
                    triangle_loss += (
                        torch.max(
                            torch.tensor(0.0, device=device),
                            dist_ij + dist_jk - dist_ik,
                        )
                        ** 2
                    )
        return triangle_loss

    def total_loss(self, alignment_data):
        """Общая функция потерь"""
        loss_align = self.alignment_loss(alignment_data)
        loss_asymmetry = self.asymmetry_loss()
        loss_triangle = self.triangle_loss()

        total_loss = (
            loss_align + self.lambda1 * loss_asymmetry + self.lambda2 * loss_triangle
        )
        return total_loss
