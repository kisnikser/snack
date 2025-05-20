import torch
from tqdm import tqdm
import torch.optim as optim
from .features import FeatureSpace
from .metric import Snack


def train_model(model, dataset_path, num_epochs=10, learning_rate=0.001):
    # Загружаем данные
    alignment_data = model.load_alignment_data(dataset_path)

    # Определение оптимизатора
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for s1, s2 in tqdm(alignment_data):
            # Подготовка данных в нужный формат (например, через DataLoader)
            optimizer.zero_grad()

            # Вызов функции потерь
            loss = model.total_loss([(s1, s2)])

            # Обратное распространение ошибки
            loss.backward()

            # Обновление параметров
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(alignment_data):.4f}"
        )


if __name__ == "__main__":
    # Укажите путь к вашему датасету BALIBASE
    dataset_path = "/home/jovyan/nkiselev/studying/bioinformatics/code/data"

    # Инициализируем пространство признаков
    feature_space = FeatureSpace()

    # Инициализируем модель с метрикой
    metric = Snack(feature_space=feature_space)

    # Запускаем обучение
    train_model(
        model=metric,
        dataset_path=dataset_path,
        num_epochs=10,
        learning_rate=0.001,
    )
