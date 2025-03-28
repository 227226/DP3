def Augmentation(info,
                 heart_center_relative_coordinates,
                 dataset_folder,
                 augmentation_factor=1,
                 control=0,  
                 bb_size=(220, 220, 200)):
    
    """
    Funkce provede augmentaci obrazů z orignálního datasetu. 

    Args:
        info (list):                                 seznam obsahující informace o obrazech
        heart_center_relative_coordinates (list):    seznam obsahující informace o poloze srdce v obraze
        dataset_folder (str):                        cesta ke složce, kde budou uloženy augmentované obrazy
        augmentation_factor (int):                   počet augmentovaných obrazů
        control (int):                               určuje, zda budou práděny kontroly:
                                                         0 - nevykresluje se nic
                                                         1 - vizualizace škálování
                                                         2 - vizualizace zanášené rotace, kontrola povolených stavů 
                                                         3 - kontrola extrahovaného srdce, které již bude úkládáno do datasetu 
        bb_size (tuple):                             velikost bounding boxu
    Returns:
        1 (int) 
    """
    

    import os 
    import json
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    import numpy as np

    from numpy.linalg import inv
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from random import sample
    from Functions import RedirectImage,\
                          ApplyPaddingZ,\
                          AffineTransform3D,\
                          SaveAsNifti,\
                          AffineMatrix,\
                          CreateBoundingBox,\
                          CubeRotation,\
                          PermittedState,\
                          AnglesVariations,\
                          ExtractCubeV2,\
                          GaussianNoise3D
    from Visualisations import CreateSlicers   

    
    # inicializace seznamu pro ukládání informací o obrazech:
    info_list = []
    # vytvoření složky pro ukládání obrazů, pokud neexistuje:
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    k = 1
    aug_size = len(info) * augmentation_factor + len(info)

    # iterace přes všechny obrazy:  
    for i in range(0, len(info)):
        # načtení transformační matice (rotace+škálování):
        A = np.array(info[i]["TransformSurvToSA"]["A"])
        A_save = info[i]["TransformSurvToSA"]["A"]
        # načtení škálovací matice:
        B = np.array(info[i]["TransformScout"]["A"])
        B_save = info[i]["TransformScout"]["A"]
        # odškálování rotační matice:
        rot_A = B @ A
        rot_A_save = rot_A.tolist()
        # vytvoření cesty k originálnímu obrazu:
        # nii_path = os.path.join(root_dir, info[i]['FolderID'], 'Scout', 's3D_BTFE_NAV.nii')
        nii_path = info[i]['pathOrigScoutnii']
        # načtení originálního obrazu:
        sitk_image = sitk.ReadImage(nii_path)
        # získání rozlišení obrazu:
        spacing = sitk_image.GetSpacing()
        # získání velikosti obrazu:
        size = sitk_image.GetSize()
        # výpočet poloměru obrazu (je nutné, aby bylo možné zajistit, že se bb nedostane mimo obraz a současně byla zajištěna 
        # dostatečná vzdálenost mezi srdcem a okrajem obrazu, což umožní zanášet větší rotace):
        x = size[0] * spacing[0] / 2
        y = size[1] * spacing[1] / 2
        z = size[2] * spacing[2] / 2
        # korekce os:    
        redirected_image = RedirectImage(sitk_image)
        # aplikace paddingu:
        padded_image = ApplyPaddingZ(redirected_image, size[0])
        # inverze škálovací matice:
        inv_B = inv(B)
        # škálování obrazu:
        scaled_image = AffineTransform3D(padded_image, inv_B, None, None)


        #===================================================== KONTROLA 1 =====================================================# 
        if control == 1:
            # extrakce pole:
            scaled_image_array = sitk.GetArrayFromImage(scaled_image)
            # vytvoření vizualizace:
            fig = CreateSlicers(scaled_image_array, manual=True)
            plt.show()
        #=====================================================================================================================# 

        # načtení relativní translace, POZOR nebude fungovat, pokud nebude provedena manuální anotace středu srdce ... 
        translation = np.array(heart_center_relative_coordinates[i])
        # korekce středu srdce, aby se bb nedostal mimo obraz, řešení okrajových podmínek:
        if translation[0] > 0:
            if translation[0] + 115 > x - 20:
                translation[0] = np.round(x, 0) - 25 - bb_size[0] // 2
        else:
            if translation[0] - 115 < -x + 20:
                translation[0] = np.round(-x, 0) + 25 + bb_size[0] // 2

        if translation[1] > 0:
            if translation[1] + 115 > y - 20:
                translation[1] = np.round(y, 0) - 25 - bb_size[1] // 2
        else:
            if translation[1] - 115 < -y + 20:
                translation[1] = np.round(-y, 0) + 25 + bb_size[1] // 2

        if translation[2] > 0:
            if translation[2] + 95 > z - 20:
                translation[2] = np.round(z, 0) - 25 - bb_size[2] // 2
        else:
            if translation[2] - 95 < -z + 20:
                translation[2] = np.round(-z, 0) + 25 + bb_size[2] // 2
        # korektrovaný střed srdce, odpovídá středu bb:
        corrected_translation = translation

        # vytvoření primárního ožezání obrazu:
        extracted_heart_original = ExtractCubeV2(scaled_image, (translation[0], translation[1], translation[2]), size=bb_size)
        extracted_heart_original_array = sitk.GetArrayFromImage(extracted_heart_original)
        extracted_heart_original_noisy = GaussianNoise3D(extracted_heart_original_array, max_value=600)

        # vytvoření informací o augmentaci:
        AugmentID = "{:03}".format(0)
        OriginalID = "{:03}".format(i)
        FolderID = AugmentID + OriginalID
        augmented_path_folder = os.path.join(dataset_folder, FolderID)

        if not os.path.exists(augmented_path_folder):
            os.makedirs(augmented_path_folder)

        augmented_path_whole = os.path.join(augmented_path_folder, 's3D_BTFE_NAV.nii.gz')

        # uložení augmentovaného srdce ve formátu NIFTI v gunzipovaném formátu:
        SaveAsNifti(extracted_heart_original_noisy, augmented_path_whole)

        augmented_info = {'ID': {'OriginalFolderID': info[i]['FolderID'], 
                                'AugmentID': AugmentID,
                                'OriginalID': OriginalID,
                                'FolderID': FolderID}, 
                        'A': {'ScalingA': B_save,
                                'OriginalA': A_save,
                                'RotationA': rot_A_save,
                                'AugmentationA': np.eye(4).tolist(),
                                'AugmentedA': rot_A_save, 
                                'Angles': (0, 0, 0)},
                        'HeartCenter': corrected_translation.tolist(),
                        'Paths': augmented_path_whole 
                        }
        
        info_list.append(augmented_info)

        print(f'Vytvoření {k}. z {aug_size} augmentací.')
        k += 1

        # vytvořění všech náhodných úhlových variací podle úhlových kroků specifikovaných v AnglesVariations:
        angles_variations = AnglesVariations()
        # inicializace seznamu pro ukládání povolených stavů:
        permitted_states = []
        # procházení všech úhlových variací:
        for angles_variation in angles_variations:
            # vytvoření Augmentační matice rotace:
            augmentation_matrix = AffineMatrix(angles=angles_variation, order_of_rotation='xyz')
            # korekce středu srdce o rotaci, trasování středu srdce:
            corrected_translation_rotated = np.round(augmentation_matrix[:3, :3] @ corrected_translation, 0)
            # vytvoření bounding boxu se středem v transformovaném středu srdce:
            vertices_bb, faces_bb = CreateBoundingBox(size=bb_size, center_point=(corrected_translation_rotated[0],
                                                                                corrected_translation_rotated[1],
                                                                                corrected_translation_rotated[2]))
            # vytvoření krychle (reprezentující původní, ale středený objem, symetrie kolem nuly, v originálním obrazu
            # není pravda, rotace ale probíhá kolem středu obrazu, stejně tak jsou potom také zanášeny úhlové inkrementy):
            vertices, faces = CreateBoundingBox(size=(int(size[0]),
                                                    int(size[1]),
                                                    int(size[2])),
                                                center_point=(0, 0, 0))
            # škálování krychle:
            vertices, faces = CubeRotation(vertices, B, rotation_center=(0, 0, 0))
            # rotaci krychle (augmentační inkrement): 
            vertices, faces = CubeRotation(vertices, augmentation_matrix, rotation_center=(0, 0, 0))
            # kontrola povoleného stavu:    
            permitted_state = PermittedState(vertices, vertices_bb, angles_variation)
            # akceptovatelné jsou pouze stavy, které obsahují alespoň jednu nenulovou rotaci:
            if permitted_state[0] == 0 and permitted_state[1] == 0 and permitted_state[2] == 0:
                continue
            else:
                permitted_states.append(permitted_state)

            #===================================================== KONTROLA 2 =====================================================# 
            if control == 2:

                fig = plt.figure(figsize=(24, 8))

                face_alpha = 0.3
                color_im = 'red'
                color_bb = 'blue'

                # První subplot - pohled zepředu (elev=90, azim=0)
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.add_collection3d(Poly3DCollection(faces, alpha=face_alpha, edgecolor=color_im, facecolor=color_im))
                ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color_im, s=50, edgecolors='k', label="Krychle 10")
                ax1.add_collection3d(Poly3DCollection(faces_bb, alpha=face_alpha, edgecolor=color_bb, facecolor=color_bb))
                ax1.scatter(vertices_bb[:, 0], vertices_bb[:, 1], vertices_bb[:, 2], color=color_bb, s=50, edgecolors='k', label="Bounding box")
                ax1.view_init(elev=270, azim=90)
                ax1.set_title('Pohled zepředu\nelev=90, azim=0')

                # Druhý subplot - pohled shora (elev=180, azim=0)
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.add_collection3d(Poly3DCollection(faces, alpha=face_alpha, edgecolor=color_im, facecolor=color_im))
                ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color_im, s=50, edgecolors='k', label="Krychle 10")
                ax2.add_collection3d(Poly3DCollection(faces_bb, alpha=face_alpha, edgecolor=color_bb, facecolor=color_bb))
                ax2.scatter(vertices_bb[:, 0], vertices_bb[:, 1], vertices_bb[:, 2], color=color_bb, s=50, edgecolors='k', label="Bounding box")
                ax2.view_init(elev=180, azim=0)
                ax2.set_title('Pohled shora\nelev=180, azim=0')

                # Třetí subplot - pohled z boku (elev=180, azim=270)
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.add_collection3d(Poly3DCollection(faces, alpha=face_alpha, edgecolor=color_im, facecolor=color_im))
                ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color_im, s=50, edgecolors='k', label="Krychle 10")
                ax3.add_collection3d(Poly3DCollection(faces_bb, alpha=face_alpha, edgecolor=color_bb, facecolor=color_bb))
                ax3.scatter(vertices_bb[:, 0], vertices_bb[:, 1], vertices_bb[:, 2], color=color_bb, s=50, edgecolors='k', label="Bounding box")
                ax3.view_init(elev=180, azim=270)
                ax3.set_title('Pohled z boku\nelev=180, azim=270')

                # Nastavení stejného měřítka os pro všechny subploty
                for ax in [ax1, ax2, ax3]:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.legend()
                    ax.set_xlim([-400, 400])
                    ax.set_ylim([-400, 400])
                    ax.set_zlim([-400, 400])

                plt.tight_layout()

                # augmentace obrazu:
                rotated_image = AffineTransform3D(scaled_image, inv(augmentation_matrix), None, None)
                # extrakce srdce:
                extracted_cube = ExtractCubeV2(rotated_image, (corrected_translation_rotated[0],
                                                            corrected_translation_rotated[1],
                                                            corrected_translation_rotated[2]), size=bb_size)
                extracted_cube_array = sitk.GetArrayFromImage(extracted_cube)
                extracted_cube = sitk.GetImageFromArray(extracted_cube_array)
                # augmentovaná matice rotace (rotace do dané srdeční roviny):
                augmented_A = augmentation_matrix @ rot_A
                # extrakce srdce po rotaci:
                extracted_cube_rotated = AffineTransform3D(extracted_cube, augmented_A, None, None)
                extracted_cube_rotated_array = sitk.GetArrayFromImage(extracted_cube_rotated)
                # extrakce srdce v originální rovině:
                rotated_image_array_reference = AffineTransform3D(scaled_image, rot_A, None, None)
                rotated_image_array_reference_array = sitk.GetArrayFromImage(rotated_image_array_reference)
                # vizualizace:  
                fig = CreateSlicers(rotated_image_array_reference_array, manual=True)
                fig = CreateSlicers(extracted_cube_array, manual=True)
                fig = CreateSlicers(extracted_cube_rotated_array, manual=True)
                plt.show()
            #=====================================================================================================================# 

        # náhodné vybrání povolených stavů (počet koresponduje s augmentačním faktorem):
        random_permitted_states = sample(permitted_states, min(augmentation_factor, len(permitted_states)))
        # pokud nebyly nalezeny žádné povolené stavy, pokračuje se v augmentaci s dalším obrazem:   
        if len(random_permitted_states) == 0:
            print("Nebyly nalezeny žádné povolené stavy. \nAugmentace pokračuje dále!")
            continue
        else:
            # procházení všech náhodně vybraných povolených stavů:
            for j, random_permitted_state in enumerate(random_permitted_states):
                # vytvoření augmentační matice rotace:
                augmentation_matrix = AffineMatrix(angles=random_permitted_state, order_of_rotation='xyz')
                augmentation_matrix_save = augmentation_matrix.tolist()
                # augmentace obrazu:
                augmented_image = AffineTransform3D(scaled_image, inv(augmentation_matrix), None, None)
                # korekce středu srdce o rotaci:
                heart_center_rotated = np.round(augmentation_matrix[:3, :3] @ corrected_translation, 0)
                # extrakce srdce:
                extracted_heart = ExtractCubeV2(augmented_image, (heart_center_rotated[0],
                                                                heart_center_rotated[1],
                                                                heart_center_rotated[2]), size=bb_size)
                extracted_heart_array = sitk.GetArrayFromImage(extracted_heart)
                extracted_heart_array_noisy = GaussianNoise3D(extracted_heart_array, max_value=600)
                # augmentovaná matice rotace (rotace do dané srdeční roviny):
                augmented_A = augmentation_matrix @ rot_A
                augmented_A_save = augmented_A.tolist()

                # vytvoření informací o augmentaci:
                AugmentID = "{:03}".format(j+1)
                OriginalID = "{:03}".format(i)
                FolderID = AugmentID + OriginalID
                augmented_path_folder = os.path.join(dataset_folder, FolderID)

                if not os.path.exists(augmented_path_folder):
                    os.makedirs(augmented_path_folder)

                augmented_path_whole = os.path.join(augmented_path_folder, 's3D_BTFE_NAV.nii.gz')

                # uložení augmentovaného srdce ve formátu NIFTI v gunzipovaném formátu:
                SaveAsNifti(extracted_heart_array_noisy, augmented_path_whole)

                augmented_info = {'ID': {'OriginalFolderID': info[i]['FolderID'], 
                                        'AugmentID': AugmentID,
                                        'OriginalID': OriginalID,
                                        'FolderID': FolderID}, 
                                'A': {'ScalingA': B_save,
                                        'OriginalA': A_save,
                                        'RotationA': rot_A_save,
                                        'AugmentationA': augmentation_matrix_save,
                                        'AugmentedA': augmented_A_save, 
                                        'Angles': random_permitted_state.tolist()},
                                'HeartCenter': heart_center_rotated.tolist(),
                                'Paths': augmented_path_whole 
                                }
        
                info_list.append(augmented_info)

                print(f'Vytvoření {k}. z {aug_size} augmentací.')
                k += 1



                #===================================================== KONTROLA 3 =====================================================# 
                if control == 3:
                    # extrakce pole:
                    extracted_heart_array = sitk.GetArrayFromImage(extracted_heart)
                    # vytvoření obrazu:
                    extracted_heart = sitk.GetImageFromArray(extracted_heart_array)
                    extracted_heart_rotated = AffineTransform3D(extracted_heart, augmented_A, None, None)
                    extracted_heart_rotated_array = sitk.GetArrayFromImage(extracted_heart_rotated)
                    # vizualizace:
                    fig = CreateSlicers(extracted_heart_array, manual=True)
                    fig = CreateSlicers(extracted_heart_rotated_array, manual=True)
                    plt.show()
                #=====================================================================================================================#     

    with open(os.path.join(dataset_folder, 'info.json'), 'w') as fi:
        json.dump(info_list, fi, indent=4)

    print("Augmentace byla dokončena.")
    return 1