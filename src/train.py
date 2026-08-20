import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import model
import dataset 
import time
import numpy as np

EPOCHS = 20
BATCH_SIZE = 64
K_FOLDS = 5
PATIENCE = 4

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

def main():
    total_time = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_ds = dataset.ArtDataset("dataset", transform=dataset.train_transform)
    val_ds = dataset.ArtDataset("dataset", transform=dataset.inference_transform)

    print("Extraindo rótulos para estratificação...")
    labels = train_ds.labels 
    
    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*20} FOLD {fold + 1}/{K_FOLDS} {'='*20}")
        
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx) 

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model_train = model.CNN().to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_train.parameters(), lr=0.001)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        early_stopping = EarlyStopping(patience=PATIENCE)

        for epoch in range(1, EPOCHS + 1):
            init_time = time.perf_counter()
            model_train.train()
            running_loss = 0.0
            
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model_train(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)

            model_train.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(device), labels_batch.to(device)
                    outputs = model_train(images)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item() * images.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
            
            val_loss /= len(val_loader.dataset)
            
            scheduler.step(val_loss)
            early_stopping(val_loss, model_train)
            
            print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            final_time = time.perf_counter()
            total_epoch_time = final_time - init_time
            total_time += total_epoch_time
            
            print()
            print("Tempo de execução de treino:", total_epoch_time, "segundos no epoch", epoch)

            if early_stopping.early_stop:
                print(f"Early stopping ativado na época {epoch}.")
                break

        print("Carregando melhores pesos do modelo para avaliação final do fold...")
        model_train.load_state_dict(early_stopping.best_model_state)
        model_train.to(device)
        model_train.eval()
        
        final_preds = []
        final_labels = []
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                outputs = model_train(images)
                _, predicted = torch.max(outputs, 1)
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels_batch.cpu().numpy())

        fold_acc = accuracy_score(final_labels, final_preds)
        fold_f1 = f1_score(final_labels, final_preds, average='macro')
        
        fold_accuracies.append(fold_acc)
        fold_f1_scores.append(fold_f1)
        
        print(f"\n--- Resultados do Fold {fold + 1} ---")
        print(f"Acurácia: {fold_acc:.4f} | F1-Score (Macro): {fold_f1:.4f}")
        print("Relatório de Classificação:")
        print(classification_report(final_labels, final_preds, zero_division=0))

    print(f"\n{'='*20} RESULTADOS FINAIS (MÉDIA DOS FOLDS) {'='*20}")
    print(f"Acurácia Média: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"F1-Score Médio: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")
    
    print()
    print("Tempo de execução total do treino: ", total_time / 60, "minutos")
    
    torch.save(model_train.state_dict(), "model_best_fold.pth")
    print("Treinamento concluído.")

if __name__ == "__main__": 
    main()