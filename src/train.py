import torch
import torch.nn as nn #modulo do pytorch para construção e treinamento de redes neurais
from torch.utils.data import DataLoader #Cria um iterável do Dataset para fácil acesso das samples
import model
import dataset
import time

EPOCHS = 20 #um epoch se refere a uma "passagem" completa através de todo dataset de treinamento, onde cada sample é rodado no modelo e seus parâmetros são atualizados com base nos erros calculados.

def main():
    train_size = int(0.8 * len(dataset.dataset))
    val_size = len(dataset.dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset.dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=40, #hiperparâmetro que determina quantos samples são processados mutuamente em que afeta a frequência de atualizações.
        shuffle=True, #mistura e embaralha os itens do array
        num_workers=4, #paralelismo
        #pin_memory=True #impulsiona a velocidade de processamento de imagens com placas da NVIDIA, mas funciona melhor com datasets maiores como 1000+
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=40, #hiperparâmetro que determina quantos samples são processados mutuamente em que afeta a frequência de atualizações.
        shuffle=True, #mistura e embaralha os itens do array
        num_workers=4, #paralelismo
        #pin_memory=True #impulsiona a velocidade de processamento de imagens com placas da NVIDIA, mas funciona melhor com datasets maiores como 1000+
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Utiliza o processamento da placa de vídeo NVIDIA ao invés do processador
    model_train = model.CNN().to(device)
    model_train.train()

    criterion = nn.CrossEntropyLoss() #calcula a perda e verifica as probabilidades da ocorrência de certos eventos em um experimento.

    optimizer = torch.optim.Adam( #algoritmo específico para otimização estocástica (maximizar ou minimizar funções objetivo na presença de incerteza)
        model_train.parameters(),
        #lr=0.001 #learning rate/taxa de aprendizado
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    init_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_train(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True) #reseta os gradientes de todo os tensores otimizados.

            loss.backward()

            optimizer.step()
            
            scheduler.step()

            running_loss += loss.item()
        
        model_train.eval()

        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model_train(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Val Loss: {val_loss:.4f}")

        model_train.train()
            
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch}/{EPOCHS} | Loss: {epoch_loss:.8f} | Acc: {accuracy:.2f}%")

    final_time = time.perf_counter()

    print("Tempo de execução de treino: ", final_time-init_time, "segundos")

    torch.save(model_train.state_dict(), "model.pth")#salva o treinamento para evitar que execute o treino em toda execução do código

if __name__ == "__main__": #para não bugar com o paralelismo
    main()