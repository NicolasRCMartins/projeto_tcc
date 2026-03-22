import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import model
import dataset
import time

EPOCHS = 20

def main():

    loader = DataLoader(
        dataset.dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4, #paralelismo
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_train = model.CNN().to(device)
    model_train.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model_train.parameters(),
        lr=0.001
    )

    init_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        running_loss = 0

        for images, labels in loader:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_train(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {epoch_loss:.8f}")

    final_time = time.perf_counter()

    print("Tempo de execução de treino: ", final_time-init_time, "segundos")

    torch.save(model_train.state_dict(), "model.pth")

if __name__ == "__main__":
    main()