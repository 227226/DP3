import os 
import json
import resnet3d

import torch 

from AugmentationScript import Augmentation
from Functions import RandomSelection
from ModelFunctions import train, test
from Opts import parse_opts
from DataLoader import NiftiDataset

if __name__ == "__main__":
   
    ###=======================PARAMETRY DEFINOVANÉ UŽIVATELEM=======================###
    # cílová velikost obrazu, ALE POZOR, jedná se o obraz, který bude prezentován modelu,
    # nikoliv velikost obrazu, který vznikne augmentací:
    target_size = 128
    # velikost bounding boxu, který bude extrahován, odpovídá velikosti obrazu, který 
    # vznikne augmentací:
    bb_size = (220, 220, 220)
    # parametr, který určuje, za má být provedena augmentace či nikoliv, NUTNO NASTAVIT PŘED SPUŠTĚNÍM SKRIPTU:
    do_augmentation = 0             
    # načtení souboru s informacemi o obrazu
    info_file = r'D:\Original\transformInfo.json'                   # musí být dodefinováno
    # načtení cesty ke složce, kde budou uloženy augmentované obrazy
    dataset_folder = r'C:\Users\42060\Desktop\DP3\DataSet'          # musí být dodefinováno 
    # definice cesty k ukládání obrázku z průběhu tréninku pro jednotlivé modely:
    picture_directory_depth = r'C:\Users\42060\Desktop\DP3\Pictures_depth' 
    picture_directory_batch = r'C:\Users\42060\Desktop\DP3\Pictures_batch' 
    # definice cesty k adresáři, kde budou uloženy modely:
    model_directory_depth = r'C:\Users\42060\Desktop\DP3\Models_depth'
    model_directory_batch = r'C:\Users\42060\Desktop\DP3\Models_batch'
    # definice parametru, který určuje, jaká bude provedena optimalizace modelu:
    optimization = 1
    ###=================================================================================###
    # načtení souboru s informacemi o transformaci obrazu
    with open(info_file, 'r') as f0:
        info = json.load(f0)
    
    # načtení souboru s relativní polohou srdce v obraze
    with open(r'heart_center_relative_coordinates.json', 'r') as f:
        heart_center_relative_coordinates = json.load(f)

    if do_augmentation == 1:
        Augmentation(info,                               # soubor obsahující ORIGINÁLNÍ informace o datasetu
                     heart_center_relative_coordinates,  # soubor nesoucí informace o RELATIVNÍ poloze srdce v obraze,
                                                         # ZÍSKANÝ MANUÁLNÍ NOTACÍ
                     dataset_folder,                     # cesta k složce, kde budou uloženy augmentované obrazy
                     augmentation_factor=1,              # určuje, kolik augmentovaných obrazů bude vytvořeno 
                     control=0,                          # určuje, zda budou práděny kontroly (nevhodné pro hlavní skript,
                                                         # proto defaultně 0)
                                                             # 0 - nevykresluje se nic
                                                             # 1 - vizualizace škálování
                                                             # 2 - vizualizace zanášené rotace, kontrola povolených stavů 
                                                             # 3 - kontrola extrahovaného srdce, které již bude úkládáno do datasetu     
                    bb_size=bb_size)            # velikost bounding boxu určující velikost extrahovaného objemu 
        
        augmented_info_path = os.path.join(dataset_folder, 'info.json')
        with open(augmented_info_path, 'r') as f1:
            augmented_info = json.load(f1)

        # rozdělení dat do trénovací, validační a testovací množiny
        RandomSelection(info,           # seznam obsahující informace originálních obrazech
                        augmented_info, # seznam obsahující informace o augmentovaných obrazech
                        dataset_folder) # cesta ke složce, kde bude uložen dataset 

    if optimization == 0:
        depth = [18, 34, 50, 101]
        for d in depth:
            opt = parse_opts()
            print("=======================================================================================")
            print("Změna hloubky sítě [18, 34, 50, 101]")
            print(f"Trénink: ResNet3D-{d} | batch_size={opt.batch_size} | target_size={target_size}")
            # inicializace parametrů:
            # inicializace parseru:
            # nastavení cesty k datasetu:
            opt.dir = dataset_folder
            # nastavení velikosti obrazu, který bude prezentován modelu (obraz by měl být vždy velikosti krychle, 
            # jistota, že bude dobře akceptován modelem a dimeze by měly být dělitelné 32):
            opt.resize = target_size/bb_size[0] 
            # nastavení cesty k souboru s informacemi o transformaci obrazu:
            transform_info_file_train = os.path.join(opt.dir, 'train_info.json')
            transform_info_file_valid = os.path.join(opt.dir, 'valid_info.json')
            transform_info_file_test = os.path.join(opt.dir, 'test_info.json')
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # vytvoření datasetů:
            train_dataset = NiftiDataset(opt.dir, transform_info_file_train, size=bb_size, resize=opt.resize)
            valid_dataset = NiftiDataset(opt.dir, transform_info_file_valid, size=bb_size, resize=opt.resize)
            test_dataset = NiftiDataset(opt.dir, transform_info_file_test, size=bb_size, resize=opt.resize)
            # načetení datasetů:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

            # generování modelu:
            model = resnet3d.generate_model(model_depth=d,
                                            n_classes=opt.n_classes)
            model.to(device)

            train(model=model,                               # načtení modelu 
                  model_directory=model_directory_depth,           # cesta pro uložení modelu
                  model_name='Resnet3D_SA',                  # jmeno modelu - struktura_rovina
                  traindataloader=train_loader,              # vložení trénovací množiny 
                  validdataloader=valid_loader,              # vložení validační množiny 
                  device=device,                             # zařízení: CUDA nebo CPU
                  num_epochs=2,                              # počet epoch 50 
                  learning_rate=opt.learning_rate,           # rychlost učení
                  mode=1,                                    # mode určuje, na základě čeho probíhá optimalizace modelu 
                                                                  # 0 - MSE loss pouze pro odhad rotační matice 
                                                                  # 1 - MSE loss tvořená odhadem rotační matice a 
                                                                  #     odhadem ortogonality matice (zajištění, že rotace 
                                                                  #     nebude zkreslena
                  alpha=1,                                   # alpha určuje váhu chyby odhadu A
                  picture_directory=picture_directory_depth,       # cesta k adresáři, kde budou uloženy obrázky z průběhu tréninku
                  b_size=opt.batch_size,                     # velikost batchu, defaultně 4
                  depth=d)                                   # hloubka modelu, mění se podle d 
    
#            avg_loss, results = test(model,         # načtení modelu 
#                                     test_loader,   # vložení testovací množiny 
#                                     device,        # zařízení: CUDA nebo CPU
#                                     mode=1)        # mode určuje, na základě čeho probíhá optimalizace modelu 
                                                        # 0 - MSE loss pouze pro odhad rotační matice 
                                                        # 1 - MSE loss tvořená odhadem rotační matice a 
                                                        #     odhadem ortogonality matice (zajištění, že rotace 
                                                        #     nebude zkreslena
            print(f"Dotrénováno: ResNet3D-{d} | batch_size={opt.batch_size} | target_size={target_size}")
            print("=======================================================================================")

    elif optimization == 1:
        b_size = [1, 2, 4]
        for b in b_size:
            opt = parse_opts()
            print("=======================================================================================")
            print("Změna velikosti batche [1, 2, 4]")
            print(f"Trénink: ResNet3D-18 | batch_size={b} | target_size={target_size}")
            # inicializace parametrů:
            # nastavení cesty k datasetu:
            opt.dir = dataset_folder
            # nastavení velikosti obrazu, který bude prezentován modelu (obraz by měl být vždy velikosti krychle, 
            # jistota, že bude dobře akceptován modelem a dimeze by měly být dělitelné 32):
            opt.resize = target_size/bb_size[0] 
            # nastavení cesty k souboru s informacemi o transformaci obrazu:
            transform_info_file_train = os.path.join(opt.dir, 'train_info.json')
            transform_info_file_valid = os.path.join(opt.dir, 'valid_info.json')
            transform_info_file_test = os.path.join(opt.dir, 'test_info.json')
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # vytvoření datasetů:
            train_dataset = NiftiDataset(opt.dir, transform_info_file_train, size=bb_size, resize=opt.resize)
            valid_dataset = NiftiDataset(opt.dir, transform_info_file_valid, size=bb_size, resize=opt.resize)
            test_dataset = NiftiDataset(opt.dir, transform_info_file_test, size=bb_size, resize=opt.resize)
            # načetení datasetů:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=b, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b, shuffle=False)

            # generování modelu:
            model = resnet3d.generate_model(model_depth=18,  # tady potom dosadit hloubku, na které bylo dosaženo nejlepšího výsledku
                                            n_classes=opt.n_classes)
            model.to(device)

            train(model=model,                                # načtení modelu
                  model_directory=model_directory_batch,            # cesta pro uložení modelu
                  model_name='Resnet3D_SA',                   # jméno modelu - struktura_rovina
                  traindataloader=train_loader,               # vložení trénovací množiny 
                  validdataloader=valid_loader,               # vložení validační množiny 
                  device=device,                              # zařízení: CUDA nebo CPU
                  num_epochs=2,                               # počet epoch 50 
                  learning_rate=opt.learning_rate,            # rychlost učení
                  mode=1,                                     # mode určuje, na základě čeho probíhá optimalizace modelu 
                                                                  # 0 - MSE loss pouze pro odhad rotační matice 
                                                                  # 1 - MSE loss tvořená odhadem rotační matice a 
                                                                  #     odhadem ortogonality matice (zajištění, že rotace 
                                                                  #     nebude zkreslena
                  alpha=1, 
                  picture_directory=picture_directory_batch,        # cesta k adresáři, kde budou uloženy obrázky z průběhu tréninku
                  b_size=b,                                   # velikost batchu, defaultně 4
                  depth=18)                                   # hloubka modelu, mění se podle d 
            print(f"Trénuji ResNet3D-{18} | batch_size={b} | target_size={target_size}")
            print("=======================================================================================")
    
#            avg_loss, results = test(model,         # načtení modelu 
#                                     test_loader,   # vložení testovací množiny 
#                                     device,        # zařízení: CUDA nebo CPU
#                                     mode=1)        # mode určuje, na základě čeho probíhá optimalizace modelu 
                                                        # 0 - MSE loss pouze pro odhad rotační matice 
                                                        # 1 - MSE loss tvořená odhadem rotační matice a 
                                                        #     odhadem ortogonality matice (zajištění, že rotace 
                                                        #     nebude zkreslena

                                            