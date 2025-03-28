import os

# import json

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
# from numpy.linalg import det
import matplotlib.pyplot as plt

# import resnet3d
# from opts import parse_opts
# from dataLoader import NiftiDataset
# from torch.utils.data import random_split


def determinant(matrix):

    """
    Funkce vypočítá determinant matice.
    Args:
        matrix (torch.tensor):          tensor o tvaru (batch_size, 3, 3)
    Returns:
        tensor o tvaru (batch_size, 1): determinant matice
    """

    return torch.linalg.det(matrix)


def is_orthogonal(matrix):

    """
    Funkce vypočítá ortogonalitu matice.
    Args:
        matrix (torch.tensor):          tensor o tvaru (batch_size, 3, 3)
    Returns:
        tensor o tvaru (batch_size, 1): chyba dopočítaná z odhadnuté ortogonální matice a
                                        identity matice
    """

    batch_size, _, _ = matrix.shape
    identity = torch.eye(3, device=matrix.device).expand(batch_size, 3, 3)
    return torch.nn.functional.mse_loss(torch.bmm(matrix.transpose(1, 2), matrix), identity)


def train(model,
          model_directory,
          model_name,
          traindataloader,
          validdataloader,
          device,
          num_epochs=30,
          learning_rate=0.0007,
          mode=0,
          alpha=1, 
          picture_directory=None,
          b_size=4,
          depth=18): # původní learning_rate: 0.001
    
    os.makedirs(model_directory, exist_ok=True)

    # setting of criterion and optimization algorithm:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # SGD
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=5)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = 10
    # initialization of supportive variables for control of error development:
    average_loss_train = []
    average_loss_valid = []


    for epoch in range(num_epochs):
        # tréninkový mód:
        model.train()
        running_batch_loss_train = 0.0
        running_epoch_loss_train = 0.0
        for i, (data, ground_truth) in enumerate(traindataloader):

            # dopředný průchod  :
            outputs = model(data.to(device))  # outputs: (batch_size=4, 3, 3)

            if mode == 0:
                # výpočet ztráty pouze pro odhad rotační matice:
                loss = criterion(outputs, ground_truth.to(device))
            elif mode == 1:
                # Ztátové funkce:
                # z predikované matice:
                loss_A = criterion(outputs, ground_truth.to(device))
                # z maticového součinu (ortogonalita):
                loss_orto = is_orthogonal(outputs.view(-1, 3, 3))  # MSE ortogonality
                # výpočet váhového faktoru gamma:
                gamma = loss_A / (loss_orto + 1e-6)
                # celková ztráta:
                loss = alpha * loss_A + gamma * loss_orto

            # Zpětná propagace, výpočet gradientu a aktualizace parametrů:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # kumulace ztrát:
            running_batch_loss_train += loss.item()
            running_epoch_loss_train += loss.item()

            if (i+1) % 4 == 0:  # Print every 5 mini-batches
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, len(traindataloader), running_batch_loss_train / 4))
                running_batch_loss_train = 0.0

        # validace:
        model.eval()
        running_epoch_loss_valid = 0.0
        with torch.no_grad():
            for j, (data, ground_truth) in enumerate(validdataloader):
                outputs = model(data.to(device))
                if mode == 0:
                    loss = criterion(outputs, ground_truth.to(device))
                elif mode == 1:
                    # Ztátové funkce:
                    # z predikované matice
                    loss_A = criterion(outputs, ground_truth.to(device))
                    # z maticového součinu (ortogonalita)
                    loss_orto = is_orthogonal(outputs.view(-1, 3, 3))  # MSE ortogonality
                    # výpočet váhového faktoru gamma:   
                    gamma = loss_A / (loss_orto + 1e-6)
                    # celková ztráta:
                    loss = alpha * loss_A + gamma * loss_orto # + beta * loss_norm

                running_epoch_loss_valid += loss.item()

        average_loss_train.append(running_epoch_loss_train/len(traindataloader)) # původně děleno i
        average_loss_valid.append(running_epoch_loss_valid/len(validdataloader)) # původně děleno j

        val_loss = running_epoch_loss_valid/len(validdataloader)
        scheduler.step(val_loss)
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(model_directory,
                                                        model_name + '_' + str(b_size) + '_' + str(depth) + '_best.pth'))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            break
        # plt.figure(1)
        # plt.plot(np.arange(1, epoch+2), np.asarray(average_loss_train))
        # plt.plot(np.arange(1, epoch+2), np.asarray(average_loss_valid))
        # plt.xlabel('Epocha')
        # plt.ylabel('Kriteriální funkce')
        # plt.show()

    if picture_directory is not None:

        os.makedirs(picture_directory, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.title(f'ResNet3D, SA: Hloubka sítě: {depth} | Velikost batche: {b_size}')
        plt.plot(np.arange(1, num_epochs + 1), np.asarray(average_loss_train), label='Trénování')
        plt.plot(np.arange(1, num_epochs + 1), np.asarray(average_loss_valid), label='Validace')
        plt.xlabel('Epocha [-]')
        plt.ylabel('Průměr MSE [-]')
        plt.xlim(0, num_epochs + 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(os.path.join(picture_directory, f'Loss_plot_{b_size}_{depth}.png'))
        # plt.show()

    # torch.save(model, model_name)
        

def test(model, testdataloader, device, mode=0, alpha=1):

    model.eval()
    criterion = nn.MSELoss()

    results = []

    total_loss = 0.0
    with torch.no_grad():
        for i, (data, ground_truth) in enumerate(testdataloader):
            outputs = model(data.to(device))

            if mode == 0:
                loss = criterion(outputs, ground_truth.to(device))
            elif mode == 1:
                # Ztátové funkce:
                # z predikované matice
                loss_A = criterion(outputs, ground_truth.to(device))
                # z maticového součinu (ortogonalita)
                loss_orto = is_orthogonal(outputs.view(-1, 3, 3))  # MSE ortogonality
                # výpočet váhového faktoru gamma:
                gamma = loss_A / (loss_orto + 1e-6)
                # celková ztráta:
                loss = alpha * loss_A + gamma * loss_orto

            total_loss += loss.item()

            # naření referenční matice a predikované matice:
            gt_matrix = ground_truth.view(-1, 3, 3).cpu().numpy()
            out_matrix = outputs.view(-1, 3, 3).cpu().numpy()

            print(f'Provedeno testování na {i+1}/{len(testdataloader)}.')

            for j in range(gt_matrix.shape[0]):
                results.append({
                    "ground_truth": gt_matrix[j].tolist(),
                    "predicted": out_matrix[j].tolist()
                })

    average_loss = total_loss / len(testdataloader)
    print(f'Průměrná chyba na testovací sadě je: {average_loss}')
    return average_loss, results