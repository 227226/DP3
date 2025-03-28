import numpy as np
import SimpleITK as sitk
import itertools
import os 
import json 

def ApplyPaddingZ(image, target_z_size):
    """
    Funkce provede padding na 3D obraz v ose Z.
    Args:
        image (sitk.Image, numpy.ndarray):        vstupní 3D obraz
        target_z_size (int):                      požadovaná velikost v ose Z
    Returns:
        padded_image (sitk.Image):                výsledný opadovaný obraz
    """

    # import SimpleITK as sitk
    # import numpy as np

    # Převod na numpy array pokud je vstup SimpleITK image
    if isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image)
        original_image = image  # Uložíme původní obraz pro metadata
    else:
        image_array = image
        original_image = sitk.GetImageFromArray(image)  # Vytvoříme SimpleITK image pro metadata
    
    # Získání současné velikosti
    current_z_size = image_array.shape[0]  # v numpy je Z první dimenze
    
    # Výpočet paddingu
    pad_z = target_z_size - current_z_size
    if pad_z <= 0:
        return original_image
    
    # Rozdělení paddingu na obě strany:
    pad_z_lower = pad_z // 2
    pad_z_upper = pad_z - pad_z_lower
    
    # Aplikace paddingu pomocí numpy
    padded_array = np.pad(image_array, 
                         ((pad_z_lower, pad_z_upper), (0, 0), (0, 0)),
                         mode='constant',
                         constant_values=0)
    
    # Převod zpět na SimpleITK image se zachováním metadat
    padded_image = sitk.GetImageFromArray(padded_array)
    
    return padded_image


def AffineTransform3D(image, transformation_matrix, center = None, translation=None):
    """
    Funkce provede výpočet transformace 3D obrazu pomocí transformační matice, zadaná transformační matice již musí být 
    inverzni. 
    Args:
        image (sitk.Image):                    vstupní 3D obraz
        transformation_matrix (numpy.ndarray): 4x4 transformační matice
        translation (bool):                    True - aplikuje i posunutí; False - aplikuje pouze rotaci
    Returns:
        rotated_image(sitkImage):              obraz po aplikaci rotace
    """

    # import SimpleITK as sitk
    # import numpy as np

    # Převod na SimpleITK image pokud je vstup numpy array:
    if isinstance(image, np.ndarray):
        sitk_image = sitk.GetImageFromArray(image)
    else:
        sitk_image = image
    
    # Výpočet centra v prostoru voxelů:
    size = sitk_image.GetSize()
    if center is None:
        center = [size[0]//2, size[1]//2, size[2]//2]
    else:
        center = center
    
    
    # Vytvoření transformace:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(transformation_matrix[:3,:3].flatten().tolist())

    if translation is not None:
        translation = translation
        transform.SetTranslation(translation)
    else:
        transform.SetTranslation([0, 0, 0])
        
    transform.SetCenter(center)
    
    # Aplikace transformace:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    
    # Výpočet transformovaného obrazu:
    rotated_image = resampler.Execute(sitk_image)
    
    return rotated_image 


def SaveAsNifti(image_array, output_path):
    """
    Uloží numpy array jako NIFTI soubor.
    Args:
        image_array (numpy.ndarray): numpy array k uložení
        output_path (str):           cesta pro uložení souboru
    """

    # import SimpleITK as sitk

    # Převod na SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_array)
    
    # Uložení souboru
    sitk.WriteImage(sitk_image, output_path)


def RedirectImage(sitk_image):
    """
    Funkce provede převrácení os obrazu tak, aby mohla probehnout rotace správně (aby byla nalezena správná srdeční rovina).
    Args:
        sitk_image (sitk.Image):     vstupní 3D obraz
    Returns:
        redirect_image (sitk.Image): obraz po redirectu
    """

    # import SimpleITK as sitk
    # import numpy as np

    # Získání směrové matice:
    direction = sitk_image.GetDirection()
    direction = np.array(direction).reshape(3, 3)

    # Vytvoření transformace:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(direction.flatten().tolist())
    transform.SetTranslation([0, 0, 0])

    # Vytvoření resamplera:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    # Výpočet redirectovaného obrazu:
    redirect_image = resampler.Execute(sitk_image)

    return redirect_image


def RestrictImage(image):

    # import SimpleITK as sitk

    size = image.GetSize()

    extractor = sitk.ExtractImageFilter()

    extractor.SetExtractionRegion([0, 0, 0, 
                                   size[0] - size[0]//2, 
                                   size[1] - size[1]//2, 
                                   size[2] - size[2]//2])
    

    restricted_image = extractor.Execute(image)
    return restricted_image


def GaussianNoise3D(img, max_value=400):
    """
    Funkce přidá do obrazu 3D Gaussovský šum s nulovou střední hodnotou a maximem kolem 400.
    Args:
        img (numpy.ndarray): numpy array k zašumění
        max_value (float): požadovaná maximální hodnota šumu (default 400)
    Returns:
        noisy_img (numpy.ndarray): obraz se zašuměním
    """
    # import numpy as np

    mean = 0
    std_dev = max_value / 3

    # generování šumu:
    gaussian_noise = np.random.normal(loc=mean, scale=std_dev, size=img.shape)

    # přidání šumu do obrazu:
    noisy_img = img + gaussian_noise

    # clipping obrazu:
    noisy_img = np.clip(noisy_img, np.min(img), np.max(img))

    return noisy_img


def AffineMatrix(angles=(0, 0, 0), order_of_rotation='xyz', scale=(1, 1, 1), scale_order=None):

    """
    Funkce provádí výpočet transformační matice pro rotaci a měřítko.
    Args:
        angles (tuple):              definice úhlů rotace kolem os x, y a z
        order_of_rotation (str):     definice pořadí aplikace rotace
        scale (tuple):               definice měřítka v osách x, y a z
        scale_order (NoneType, str): definice, zda se měřítko aplikuje před nebo po rotaci
    Returns:
        A_final (numpy.ndarray):     transformační matice 4x4
    """

    # import numpy as np

    # převod stupňů na radiány:
    alpha, beta, gamma = np.deg2rad(np.array(angles))

    # definice měřítka:
    A_scale = np.array([
                        [scale[0], 0, 0],
                        [0, scale[1], 0],
                        [0, 0, scale[2]]
                       ])
    # definice rotace kolem osy x:
    Ax = np.array([
                   [1,             0,              0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha),  np.cos(alpha)]
                  ])
    # definice rotace kolem osy y:
    Ay = np.array([
                   [ np.cos(beta),  0,   np.sin(beta)],
                   [ 0,             1,              0],
                   [-np.sin(beta),  0,   np.cos(beta)]
                  ])
    # definice rotace kolem osy z:
    Az = np.array([
                   [np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma),  np.cos(gamma), 0],
                   [            0,              0, 1]
                  ])

    # definice pořadí rotace podle zadání uživatele nebo výchozí nastavení:
    if order_of_rotation == 'xyz':
        A = Az @ Ay @ Ax
    elif order_of_rotation == 'xzy':
        A = Ay @ Az @ Ax
    elif order_of_rotation == 'yxz':
        A = Az @ Ax @ Ay
    elif order_of_rotation == 'yzx':
        A = Ax @ Az @ Ay
    elif order_of_rotation == 'zxy':
        A = Ay @ Ax @ Az
    elif order_of_rotation == 'zyx':
        A = Ax @ Ay @ Az

    # výpočet konečné transformační matice, měřítko je aplikováno podle nastavení uživatele nebo
    # není aplikováno, pokud je None:
    if scale_order == 'before':
        A_final = A @ A_scale
    elif scale_order == 'after':
        A_final = A_scale @ A
    elif scale_order == None:
        A_final = A

    A_4x4 = np.eye(4)
    A_4x4[:3,:3] = A_final

    # konečná transformační matice jako výstup funkce:
    return A_4x4


def EulerAngles(A, order_of_rotation='xyz', decimals=5):

    """
    Funkce rozloží transformační matici A na Eulerovy úhly, v daném pořadí s danou přesností v decimálách.

    Args:
        A (numpy.ndarray):                  transformační matice 4x4, pokud není matice rotace, výsledek nebude správný
        order_of_rotation (string):         pořadí dekompozice, existují šest možností pořadí dekompozice:
                                                1. xyz (default), 2. xzy, 3. yxz, 4. yzx, 5. zxy, 6. zyx
        decimals (int):                     počet desetinných míst výstupního úhlu

    Returns:
        angles (numpy.ndarray):             výstupní pole s výslednými úhly v pořadí rotace kolem os x, y a z
    """

    # import numpy as np

    # rozložení transformační matice na jednotlivé čísla pro následné výpočty:
    A11, A12, A13 = A[0, :3]
    A21, A22, A23 = A[1, :3]
    A31, A32, A33 = A[2, :3]

    # odhad úhlů dle pořadí dekompozice, výchozí pořadí 'xyz':
    if order_of_rotation == 'xyz':
        alpha = np.arctan2(A32, A33)
        beta = np.arctan2(-A31, np.sqrt(A21 ** 2 + A11 ** 2))
        gamma = np.arctan2(A21, A11)
    elif order_of_rotation == 'xzy':
        alpha = np.arctan2(-A23, A22)
        beta = np.arctan2(-A31, A11)
        gamma = np.arctan2(A21, np.sqrt(A22 ** 2 + A23 ** 2))
    elif order_of_rotation == 'yxz':
        alpha = np.arctan2(A32, np.sqrt(A31 ** 2 + A33 ** 2))
        beta = np.arctan2(-A31, A33)
        gamma = np.arctan2(-A12, A22)
    elif order_of_rotation == 'yzx':
        alpha = np.arctan2(A32, A22)
        beta = np.arctan2(A13, A11)
        gamma = np.arctan2(-A12, np.sqrt(A22 ** 2 + A32 ** 2))
    elif order_of_rotation == 'zxy':
        alpha = np.arctan2(-A23, np.sqrt(A21 ** 2 + A22 ** 2))
        beta = np.arctan2(A13, A33)
        gamma = np.arctan2(A21, A22)
    elif order_of_rotation == 'zyx':
        alpha = np.arctan2(-A23, A33)
        beta = np.arctan2(A13, np.sqrt(A23 ** 2 + A33 ** 2))
        gamma = np.arctan2(-A12, A11)

    # vytvoření pole s výslednými úhly v pořadí rotace kolem os x, y a z:
    angles = np.array([np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)])
    angles = np.round(angles, decimals)

    # výstupní pole s výslednými úhly v pořadí rotace kolem os x, y, z:
    return angles


def plane_from_points(P1, P2, P3):

    """
    Funkce vypočítá rovnici roviny ze tří zadaných bodů.

    Args:
        P1, P2, P3 (int):    body ležící v rovině

    Returns:
        A, B, C, D (float):  konstanty rovnice roviny
    """

    # import numpy as np

    # Směrové vektory
    v1 = np.array(P2) - np.array(P1)
    v2 = np.array(P3) - np.array(P1)

    # Normálový vektor (vektorový součin)
    normal = np.cross(v1, v2)
    A, B, C = normal

    # Výpočet D ze vzorce roviny
    D = -np.dot(normal, P1)

    return A, B, C, D


def PermittedState(points, bb_points, angles=(0, 0, 0)):

    """
    Funkce zkontroluje, zda se celý objem obálky nachází uvnitř původního otočeného objemu. Pokud ne, vrátí se
    (0, 0, 0) tuple a je odfiltrován v dalším bloku. Pouze nenulové rotace jsou ověřovány.

    Args:
        points (nd.array):    matice s koordinaty určujícími umístění rohů původního objemu
        bb_points (nd.array): koordinaty rohů obálky
        angles (tuple):       je vrácena i v případě, že obálka leží uvnitř původního otočeného objemu

    Returns:
        angles (tuple):       výstupní proměnná s úhly rotace
    """

    # rozložení pole na jednotlivé body:
    P1 = points[0, :]
    P2 = points[1, :]
    P3 = points[2, :]
    P4 = points[3, :]
    P5 = points[4, :]
    P6 = points[5, :]
    P7 = points[6, :]
    # P8 = points[7, :]

    # kontrola je provedena dosazením bodů do rovnice roviny, pro každou rovinu je specifikováno, kde bod obálky
    # musí ležet, pouze pokud jsou splněny všechny podmínky, jsou vráceny úhly rotace, v opačném případě jsou vrácena
    # pouze nuly:
    # levá strana:
    A1, B1, C1, D1 = plane_from_points(P1, P2, P3)
    for i in range(bb_points.shape[0]):
        in_box = A1 * bb_points[i, 0] + B1 * bb_points[i, 1] + C1 * bb_points[i, 2] + D1
        if in_box > 0:
            continue
        else:
            return (0, 0, 0)

    # pravá strana:
    A2, B2, C2, D2 = plane_from_points(P5, P6, P7)
    for i in range(bb_points.shape[0]):
        in_box = A2 * bb_points[i, 0] + B2 * bb_points[i, 1] + C2 * bb_points[i, 2] + D2
        if in_box < 0:
            continue
        else:
            return (0, 0, 0)

    # přední strana:
    A3, B3, C3, D3 = plane_from_points(P3, P4, P7)
    for i in range(bb_points.shape[0]):
        in_box = A3 * bb_points[i, 0] + B3 * bb_points[i, 1] + C3 * bb_points[i, 2] + D3
        if in_box < 0:
            continue
        else:
            return (0, 0, 0)

    # zadní strana:
    A4, B4, C4, D4 = plane_from_points(P1, P2, P6)
    for i in range(bb_points.shape[0]):
        in_box = A4 * bb_points[i, 0] + B4 * bb_points[i, 1] + C4 * bb_points[i, 2] + D4
        if in_box < 0:
            continue
        else:
            return (0, 0, 0)

    # horní strana:
    A5, B5, C5, D5 = plane_from_points(P2, P3, P6)
    for i in range(bb_points.shape[0]):
        in_box = A5 * bb_points[i, 0] + B5 * bb_points[i, 1] + C5 * bb_points[i, 2] + D5
        if in_box < 0:
           continue
        else:
           return (0, 0, 0)

    # spodní strana:
    A6, B6, C6, D6 = plane_from_points(P1, P4, P5)
    for i in range(bb_points.shape[0]):
        in_box = A6 * bb_points[i, 0] + B6 * bb_points[i, 1] + C6 * bb_points[i, 2] + D6
        if in_box > 0:
            continue
        else:
            return (0, 0, 0)

    return angles


def CubeRotation(points, A, rotation_center=(0, 0, 0)):

    """
    Funkce provádí rotaci rohů krychle o zadanou rotaci definovanou rotací matice A a středem rotace.

    Args:
        points (nd.array): rohy krychle
        A (nd.array): rotace matice
        rotation_center (tuple): střed rotace

    Returns:
        rotated_points (nd.array): rohy krychle
        rotated_faces (list): stěny krychle
    """

    # import numpy as np

    # vytvoření augmentované souřadnice (1 je přidaná):
    points_aug = np.zeros((points.shape[0], points.shape[1]+1))
    points_aug[:, 3] = 1
    points_aug[:, :3] = points

    # inicializace proměnné pro ukládání otočených rohů:
    rotated_points = np.zeros(points_aug.shape)

    # vytvoření augmentované rotace matice:
    A0 = np.eye(4)
    A0[:4, :4] = A

    # vytvoření augmentované translace matice:
    Tf = np.eye(4)
    Tb = np.eye(4)

    # přidání translace:
    rotation_center = np.array(rotation_center)
    Tf[:3, 3] = rotation_center
    Tb[:3, 3] = -rotation_center

    # konečná transformační matice:
    Af = Tf @ A0 @ Tb

    # otočení rohů:
    for i in range(points.shape[0]):
        rotated_points[i, :] = Af @ points_aug[i, :]

    rotated_points = rotated_points[:, :3]

    rotated_faces = [
             [rotated_points[0], rotated_points[1], rotated_points[2], rotated_points[3]], # přední stěna
             [rotated_points[4], rotated_points[5], rotated_points[6], rotated_points[7]], #  zadní stěna
             [rotated_points[0], rotated_points[4], rotated_points[5], rotated_points[1]], #   levá stěna
             [rotated_points[3], rotated_points[7], rotated_points[6], rotated_points[2]], #  pravá stěna
             [rotated_points[0], rotated_points[4], rotated_points[7], rotated_points[3]], #  dolní stěna
             [rotated_points[1], rotated_points[5], rotated_points[6], rotated_points[2]]  #  horní stěna
                    ]

    return rotated_points, rotated_faces


def CreateBoundingBox(size=(200, 200, 150), center_point=(0, 0, 0)):

    """
    Funkce vrátí bounding box s danou velikostí, respektive rohy bounding boxu a jeho stěny.

    Args:
        size (tuple):         velikost bounding boxu
        center_point (tuple): určuje přesné umístění bounding boxu v souřadném systému

    Returns:
        bb (nd.array):        rohy bounding boxu
        bb_faces (list):      stěny bounding boxu
    """

    # import numpy as np

    # definice rohů bounding boxu:
    bb = np.array([
                              [0      ,       0,       0],
                              [0      ,       0, size[2]],
                              [size[0],       0, size[2]],
                              [size[0],       0,       0],
                              [0      , size[1],       0],
                              [0      , size[1], size[2]],
                              [size[0], size[1], size[2]],
                              [size[0], size[1],       0]
                                                         ], dtype=np.float64)
    # centrování:
    bb[:, 0] += center_point[0] - size[0] / 2
    bb[:, 1] += center_point[1] - size[1] / 2
    bb[:, 2] += center_point[2] - size[2] / 2
    # definice stěn:
    bb_faces = [
                [bb[0], bb[1], bb[2], bb[3]],
                [bb[4], bb[5], bb[6], bb[7]],
                [bb[0], bb[4], bb[5], bb[1]],
                [bb[3], bb[7], bb[6], bb[2]],
                [bb[0], bb[3], bb[7], bb[4]],
                [bb[1], bb[5], bb[6], bb[2]]
                                            ]
    return bb, bb_faces


def AnglesVariations():

    """
    Funkce vrátí všechny možné úhlové rotace podél každé osy podle povolených rotací.

    Returns:
        rotation_variations (nd.array): všechny možné úhlové rotace
    """

    # import numpy as np
    # import itertools

    # permitted rotations along each axis:
    angles_z = [15, -15, 30, -30, 45, -45]  # Rotace kolem osy X
    angles_y = [0, 10, -10, 15, -15]  # Rotace kolem osy Y
    angles_x = [0, 10, -10, 15, -15]  # Rotace kolem osy Z
    # variation of all possible rotations:
    rotation_variations = list(itertools.product(angles_x, angles_y, angles_z))
    # conversion to nd.array
    rotation_variations = np.array(rotation_variations)

    return rotation_variations


def ExtractCube(image, size=(220, 220, 190)):
    """
    Extrahuje krychli o zadané velikosti ze středu obrazu
    
    Args:
        image: SimpleITK obraz
        size: tuple (x, y, z) velikosti extrahované krychle
    """
    # import SimpleITK as sitk

    # Získáme velikost původního obrazu
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    
    # Vypočítáme střed obrazu
    center_x = original_size[0] // 2
    center_y = original_size[1] // 2
    center_z = original_size[2] // 2
    
    # Vypočítáme počáteční index pro výřez
    start_x = center_x - size[0] // 2
    start_y = center_y - size[1] // 2
    start_z = center_z - size[2] // 2
    
    # Vytvoříme extraktor
    extractor = sitk.ExtractImageFilter()
    
    # Nastavíme velikost a počáteční index výřezu
    extraction_size = [size[0], size[1], size[2]]
    extraction_index = [start_x, start_y, start_z]
    
    extractor.SetSize(extraction_size)
    extractor.SetIndex(extraction_index)
    
    # Provedeme výřez
    extracted_image = extractor.Execute(image)
    
    # Upravíme origin pro nový obraz
    new_origin = [
        original_origin[0] + start_x,
        original_origin[1] + start_y,
        original_origin[2] + start_z
    ]
    
    extracted_image.SetOrigin(new_origin)
    
    return extracted_image


def ExtractCubeV2(image, relative_center, size=(220, 220, 190)):
    """
    Extrahuje krychli o zadané velikosti kolem zadaného relativního středu
    
    Args:
        image: SimpleITK obraz
        relative_center: tuple (x, y, z) relativní pozice středu vzhledem ke středu vstupního obrazu
        size: tuple (x, y, z) velikosti extrahované krychle
    """
    # import SimpleITK as sitk

    # Získáme velikost původního obrazu
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    
    # Vypočítáme střed obrazu
    center_x = original_size[0] // 2
    center_y = original_size[1] // 2
    center_z = original_size[2] // 2
    
    # Vypočítáme absolutní střed výřezu přičtením relativní pozice
    absolute_center_x = center_x + int(relative_center[0])
    absolute_center_y = center_y + int(relative_center[1])
    absolute_center_z = center_z + int(relative_center[2])
    
    # Vypočítáme počáteční index pro výřez
    start_x = absolute_center_x - size[0] // 2
    start_y = absolute_center_y - size[1] // 2
    start_z = absolute_center_z - size[2] // 2
    
    # Vytvoříme extraktor
    extractor = sitk.ExtractImageFilter()
    
    # Nastavíme velikost a počáteční index výřezu
    extraction_size = [size[0], size[1], size[2]]
    extraction_index = [start_x, start_y, start_z]
    
    extractor.SetSize(extraction_size)
    extractor.SetIndex(extraction_index)
    
    # Provedeme výřez
    extracted_image = extractor.Execute(image)

    # Upravíme origin pro nový obraz
    new_origin = [
        original_origin[0] + start_x,
        original_origin[1] + start_y,
        original_origin[2] + start_z
    ]
    
    extracted_image.SetOrigin(new_origin)
    
    return extracted_image


def RandomSelection(info,
                    augmented_info,
                    dataset_folder, 
                    ratios=(0.7, 0.1, 0.2),
                    random_seed=42):
    
    """
    Funkce provede náhodné rozdělení datové sady do trénovací, validační a testovací množiny
    v zadaném poměru. Aby byla zajištěna reprodukovatelnost, je zadán random seed. 
    Kromě uvedeného funkce kontroluje, zda byly všechny prvky zařazeny do množin, a že
    se žádný prvek neopakuje ve více množinách, aby bylo zamezeno datovému průniku.

    Args:
        info (list):           seznam obsahující informace originálních obrazech
        augmented_info (list): seznam obsahující informace o augmentovaných obrazech
        dataset_folder (str):  cesta ke složce, kde bude uložen dataset 
        ratios (tuple):        poměr mezi trénovací, validační a testovací množinou   
        random_seed (int):     seed pro generování náhodných čísel, reprodukovatelnost
    Returns:
        1 (int)
    """

    # import numpy as np 
    # import json 
    # import os 

    # zjištení délky původní datové sady: 
    selection_samples = list(range(len(info)))
    # nastavení parametru random seedu pro reprodukovatelnost:
    np.random.seed(random_seed)
    # náhodné promíchání prvků, vzhledem k random seed lze mezi jednotlivými spuštěními generovat stejné dělení:
    np.random.shuffle(selection_samples)
  
    # zjištění délky původní datové sady: 
    n_samples = len(selection_samples)
    # výpočet počtu prvků v jednotlivých množinách:
    n_train = int(np.floor(n_samples * ratios[0]))
    n_test = int(np.floor(n_samples * ratios[2]))
    n_val = n_samples - n_train - n_test
    # rozdělení prvků do jednotlivých množin:
    train_samples = selection_samples[:n_train]
    val_samples = selection_samples[n_train:n_train + n_val]
    test_samples = selection_samples[n_train + n_val:]
    # kontrola, zda jsou všechny prvky zařazeny do množin:
    if len(selection_samples) != len(train_samples) + len(val_samples) + len(test_samples):
        print("Některý z prvů není zařazen do žádné skupiny!")
    else:
        print("Všechny prvky byly zařazeny do množin.")
    # převedení prvků do množin pro jednodušší kontrolu:    
    train_set = set(train_samples)
    val_set = set(val_samples)
    test_set = set(test_samples)
    # ověření, že žádné dva prvky nejsou více množinách:
    assert len(train_set.intersection(val_set)) == 0, 'Překryv prvků mezi trénovací a validační množinou'
    assert len(train_set.intersection(test_set)) == 0, 'Překryv prvků mezi trénovací a testovací množinou'
    assert len(val_set.intersection(test_set)) == 0, 'Překryv prvků mezi validační a testovací množinou'
    # inicializace pro ukládání informací o obrazech z jednotlivých množin:
    train_info = []
    valid_info = []
    test_info = []
    # rozdělení informací o obrazech do jednotlivých množin:
    for sample_info in augmented_info: 
        if int(sample_info['ID']['OriginalID']) in train_samples:
            train_info.append(sample_info)
        elif int(sample_info['ID']['OriginalID']) in val_samples:
            valid_info.append(sample_info)
        elif int(sample_info['ID']['OriginalID']) in test_samples:
            test_info.append(sample_info)
    # ukládání informací o obrazech do jednotlivých množin: 
    with open(os.path.join(dataset_folder, 'train_info.json'), 'w') as f0:
        json.dump(train_info, f0, indent=4)

    with open(os.path.join(dataset_folder, 'valid_info.json'), 'w') as f1:
        json.dump(valid_info, f1, indent=4)

    with open(os.path.join(dataset_folder, 'test_info.json'), 'w') as f2:
        json.dump(test_info, f2, indent=4)

    print("Dělení do množin proběhlo úspěšně.")
    return 1