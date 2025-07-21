import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from models import SNNExoplanetDetector, LSTMExoplanetDetector
from data_utils import preprocess_light_curve
import numpy as np

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device):
    model.eval()
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_outputs)

def calculate_metrics(labels, outputs):
    preds = np.argmax(outputs, axis=1)
    auc = roc_auc_score(labels, outputs[:, 1])
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    return auc, f1, precision, recall

def main(train_loader, val_loader, input_features, hidden_features, output_features, device):
    model = SNNExoplanetDetector(input_features, hidden_features, output_features).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Example: 10 epochs
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        
        labels, outputs = evaluate_model(model, val_loader, device)
        auc, f1, precision, recall = calculate_metrics(labels, outputs)
        print(f'Evaluation - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')