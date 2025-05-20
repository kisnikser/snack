import os
from Bio import AlignIO


def load_msf_data(dataset_path):
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
