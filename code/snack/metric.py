import os
import torch
import torch.nn as nn
import numpy as np
from Bio import AlignIO


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

    def __call__(self, i, j):
        """Вычисляем расстояние между аминокислотами a_i и a_j"""
        x_i = self.feature_space.get_features(i)
        x_j = self.feature_space.get_features(j)
        delta_ij = x_i - x_j
        distance = torch.matmul(delta_ij, torch.matmul(self.M, delta_ij))
        normalized_distance = torch.tanh(distance)
        return normalized_distance

    def load_alignment_data(self, dataset_path):
        """Загружаем данные из датасета в формате MSF"""
        alignment_data = []

        # Прочитаем все .msf файлы из директории
        for file_name in os.listdir(dataset_path):
            if file_name.endswith(".msf"):
                # Загружаем файл
                alignment = AlignIO.read(os.path.join(dataset_path, file_name), "msf")
                # Извлекаем первые две последовательности
                seq1 = str(alignment[0].seq)  # Первая последовательность
                seq2 = str(alignment[1].seq)  # Вторая последовательность
                alignment_data.append((seq1, seq2))

        return alignment_data

    def alignment_loss(self, alignment_data):
        """Функция ошибки для выравнивания"""
        total_loss = 0
        for s1, s2 in alignment_data:
            # Используем Needleman-Wunsch для получения выравнивания
            aligned_seq1, aligned_seq2 = self.needleman_wunsch(s1, s2)

            # Теперь сравниваем с эталонным выравниванием
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

    def needleman_wunsch(self, seq1, seq2):
        """Алгоритм Needleman-Wunsch для выравнивания"""
        len1 = len(seq1)
        len2 = len(seq2)

        # Инициализация матрицы
        dp = np.zeros((len1 + 1, len2 + 1))

        # Заполнение первой строки и первого столбца
        for i in range(1, len1 + 1):
            dp[i][0] = dp[i - 1][0] + self(seq1[i - 1], "-")
        for j in range(1, len2 + 1):
            dp[0][j] = dp[0][j - 1] + self("-", seq2[j - 1])

        # Заполнение остальной матрицы
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                match = dp[i - 1][j - 1] + self(seq1[i - 1], seq2[j - 1])
                delete = dp[i - 1][j] + self(seq1[i - 1], "-")
                insert = dp[i][j - 1] + self("-", seq2[j - 1])
                dp[i][j] = min(match, delete, insert)

        # Восстановление выравнивания
        aligned_seq1 = []
        aligned_seq2 = []
        i, j = len1, len2
        while i > 0 and j > 0:
            if dp[i][j] == dp[i - 1][j - 1] + self(seq1[i - 1], seq2[j - 1]):
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i - 1][j] + self(seq1[i - 1], "-"):
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append("-")
                i -= 1
            else:
                aligned_seq1.append("-")
                aligned_seq2.append(seq2[j - 1])
                j -= 1

        while i > 0:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1

        while j > 0:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

        return "".join(reversed(aligned_seq1)), "".join(reversed(aligned_seq2))

    def asymmetry_loss(self):
        """Функция ошибки для асимметрии"""
        asymmetry_loss = 0
        for i in range(20):
            for j in range(i + 1, 20):
                loss = torch.abs(self(i, j) - self(j, i))
                asymmetry_loss += loss
        return asymmetry_loss

    def triangle_loss(self):
        """Функция ошибки для нарушения неравенства треугольника"""
        triangle_loss = 0
        for i in range(20):
            for j in range(i + 1, 20):
                for k in range(j + 1, 20):
                    dist_ij = self(i, j)
                    dist_jk = self(j, k)
                    dist_ik = self(i, k)
                    triangle_loss += (
                        torch.max(torch.tensor(0.0), dist_ij + dist_jk - dist_ik) ** 2
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
