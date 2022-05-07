import os, torch, datetime, warnings
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from dataset import DataGenerator
from model import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from colorama import Fore


train_path = '.\\data\\train_npy'


def metrice(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-7))) * 100
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return rmse, mape, r2, mae


def getData():
    data_list = np.array(os.listdir(train_path))
    with open('label_cap.txt') as f:
        y_csv = list(map(lambda x: x.strip().split(','), f.readlines()))
    y_label = {}
    for i in y_csv:
        try:
            y_label[i[0]] = i[1]
        except:
            continue
    x, y = [], []
    for i in data_list:
        if i[:-4] in y_label:
            x.append(train_path + '\\{}'.format(i))
            y.append(float(y_label[i[:-4]]))
    x, y = np.array(x), np.array(y)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=5)

    return x_train, x_val, y_train, y_val


if __name__ == "__main__":

    BATCH_SIZE = 512
    EPOCHS = 200

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter('runs\\full_34')

    x_train, x_val, y_train, y_val = getData()
    train_dataset = DataGenerator(x_train, y_train)
    train_iter = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=6)
    val_dataset = DataGenerator(x_val, y_val)
    val_iter = DataLoader(val_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=6)

    model = resnet18().to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0015)
    loss = torch.nn.MSELoss()

    if torch.cuda.device_count() > 1:
      print("\nLet's use", torch.cuda.device_count(), "GPUs!")
      model = nn.DataParallel(model)
    model.to(DEVICE)
    best_loss = 0x7fffffff
    print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    history = []
    for epoch in range(EPOCHS):
        with tqdm(
                desc='epoch {}'.format(epoch),
                iterable=train_iter,
                bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET),
        ) as t:
            model.train()
            train_loss, train_mae, train_rmse, train_mape, train_r2 = 0, 0, 0, 0, 0
            begin = time.time()
            for x, y in train_iter:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x.float()).squeeze()
                l = loss(pred, y.float())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_loss += float(l.data)
                rmse_v, mape_v, r2_v, mae_v = metrice(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
                train_rmse += rmse_v
                train_r2 += r2_v
                train_mae += mae_v
                t.update()
            train_loss /= len(train_iter)
            train_rmse /= len(train_iter)
            train_r2 /= len(train_iter)
            train_mae /= len(train_iter)

            val_loss, val_mae, val_rmse, val_r2 = 0, 0, 0, 0
            model.eval()
            with torch.no_grad():
                for x, y in val_iter:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x.float()).squeeze()
                    l = loss(pred, y.float())
                    val_loss += float(l.data)
                    rmse_v, mape_v, r2_v, mae_v = metrice(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
                    val_rmse += rmse_v
                    val_r2 += r2_v
                    val_mae += mae_v
            val_loss /= len(val_iter)
            val_rmse /= len(val_iter)
            val_r2 /= len(val_iter)
            val_mae /= len(val_iter)

            writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            writer.add_scalars('rmse', {'train_rmse': train_rmse, 'val_rmse': val_rmse}, epoch)
            writer.add_scalars('r2', {'train_r2': train_r2, 'val_r2': val_r2}, epoch)
            writer.add_scalars('mae', {'train_mae': train_mae, 'val_mae': val_mae}, epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model, 'model_34w.pht'.format(val_loss))
                print('\n{} save best_val_loss model success!'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            print(
                '\n{} epoch:{}, time:{:.2f}s, train_loss:{:.4f}, val_loss:{:.4f}, train_rmse:{:.4f}, val_rmse:{:.4f}, train_r2:{:.4f}, val_r2:{:.4f}, train_mae:{:.4f}, val_mae:{:.4f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch + 1, time.time() - begin, train_loss, val_loss, train_rmse, val_rmse, train_r2,
                    val_r2, train_mae, val_mae
                ))

    writer.close()