import os
from sklearn.metrics import roc_auc_score
from model import GATModel
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import TabularDataset, DailyBatchSampler





if __name__ == "__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    path = "/home/sdh/quant/DNN/dataset"
    save_dir = "/home/sdh/quant/DNN/Neutral_Network/checkpoints"
    label = pd.read_parquet(os.path.join(path, "label.parquet"))

    train_data_range = pd.date_range("2013-01-01", "2018-01-01", freq="1M").strftime("%Y-%m-01") 
    valid_data_range = pd.date_range("2018-01-01", "2020-01-01", freq="1M").strftime("%Y-%m-01")
    test_data_range = pd.date_range("2020-01-01", "2023-08-01", freq="1M").strftime("%Y-%m-01") 

    x_train = [pd.read_pickle(os.path.join(path, "factor", f"{period}.pkl")).astype("float16")
               for period in train_data_range]
    x_train = pd.concat(x_train).fillna(0)
    y_train = label.loc[x_train.index].fillna(0) 

    x_valid = [pd.read_pickle(os.path.join(path, "factor", f"{period}.pkl")).astype("float16")
               for period in valid_data_range]
    x_valid = pd.concat(x_valid).fillna(0)
    y_valid = label.loc[x_valid.index].fillna(0)

    x_test = [pd.read_pickle(os.path.join(path, "factor", f"{period}.pkl")).astype("float16")
               for period in test_data_range]
    x_test = pd.concat(x_test).fillna(0)
    y_test = label.loc[x_test.index].fillna(0)
    
    train_dataset = TabularDataset(x_train, y_train)
    valid_dataset = TabularDataset(x_valid, y_valid)
    test_dataset = TabularDataset(x_test, y_test)

    train_sampler = DailyBatchSampler(train_dataset)
    valid_sampler = DailyBatchSampler(valid_dataset)
    test_sampler = DailyBatchSampler(test_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, num_workers=10, drop_last=True)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, num_workers=10, drop_last=True)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, num_workers=10, drop_last=True)
    

    model = GATModel(input_dim=1447, hidden_size=128).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss() 
    
    best_auc = 0.
    for epoch in range(1, 10):
        print(f'epoch: {epoch}')
        # training
        model.train()
        train_loss = 0.
        for feature, label in train_loader:
            feature = feature.squeeze().to(device) # 2171 x 1447
            label = label.squeeze().float().to(device) # 2171
        
            pred = model(feature.float())
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * label.size(0)
        print(f'train loss: {(train_loss / len(train_loader)):.4f}\t')
        

        # validation
        model.eval()
        valid_loss = 0.
        predictions, labels = [], []
        with torch.no_grad():
            for feature, label in valid_loader:
                feature = feature.squeeze().to(device) 
                label = label.squeeze().to(device) 
                pred = model(feature.float())
                loss = loss_fn(pred, label.float())
                valid_loss += loss.item() * label.size(0)
                predictions.extend(torch.sigmoid(pred).cpu().numpy())
                labels.extend(label.cpu().numpy())

            valid_auc = roc_auc_score(labels, predictions)
            print(f'valid loss: {(valid_loss / len(valid_loader)):.4f}\t'
                    f'valid auc : {str(round(valid_auc, 4))}')

        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(model, os.path.join(save_dir, 'best_gat_model.pt'))

    # testing  
    # model = torch.load(os.path.join(save_dir, 'best.pt'), map_location=device)
    torch.cuda.empty_cache()
    model.eval()
    test_loss = 0.
    predictions, labels = [], []
    with torch.no_grad():
        for feature, label in test_loader:
            feature = feature.squeeze().to(device)
            label = label.squeeze().float().to(device) 
            pred = model(feature.float())
            loss = loss_fn(pred, label)
            test_loss += loss.item() * label.size(0)
            predictions.extend(torch.sigmoid(pred).cpu().numpy())
            labels.extend(label.cpu().numpy())

        test_auc = roc_auc_score(labels, predictions)
        print(f'test loss: {(test_loss / len(test_loader)):.4f}\t'
                f'test auc : {str(round(test_auc, 4))}')



