import os 
import json
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from functions import RedirectImage,\
                      ApplyPaddingZ,\
                      AffineTransform3D,\
                      SaveAsNifti,\
                      AffineMatrix,\
                      EulerAngles

from visualisations import CreateSlicers, create_interactive_coordinate_picker

### Kód pro rotaci obrazu
root_dir = r'D:\Original'
info_file = r'D:\Original\transformInfo.json'

with open(info_file, 'r') as f:
    info = json.load(f)

control = 0 

# i = 7 # 2, 5, 6

heart_center_relative_coordinates = []

for i in range(len(info)):

    A = np.array(info[i]["TransformSurvToSA"]["A"])

    B = np.array(info[i]["TransformScout"]["A"])

    nii_path = os.path.join(root_dir, info[i]['FolderID'], 'Scout', 's3D_BTFE_NAV.nii')

    sitk_image = sitk.ReadImage(nii_path)

    size = sitk_image.GetSize()

    redirected_image = RedirectImage(sitk_image)

    padded_image = ApplyPaddingZ(redirected_image, size[0])

    inv_B = inv(B)

    rotated_image = AffineTransform3D(padded_image, inv_B, None, None)

    if control == 1:
        rot_A = B @ A

        rotated_image1 = AffineTransform3D(rotated_image, rot_A, None, None)

        rotated_image1 = sitk.GetArrayFromImage(rotated_image1)

    coordinates = create_interactive_coordinate_picker(rotated_image)
    heart_center = np.array((coordinates['x'], coordinates['y'], coordinates['z']))

    image_center = np.array(rotated_image.GetSize()) / 2


    translation = heart_center - image_center
    translation = translation.tolist()
    print(translation)
    break 
    heart_center_relative_coordinates.append(translation)

    if control == 1:
        rotated_image = AffineTransform3D(rotated_image, np.eye(4), center=None, translation=translation)

        rotated_image = AffineTransform3D(rotated_image, rot_A, center=None, translation=None)

        rotated_image = sitk.GetArrayFromImage(rotated_image)

        fig = CreateSlicers(rotated_image1, manual=True)
        fig = CreateSlicers(rotated_image, manual=True)
        plt.show()

with open('heart_center_relative_coordinates.json', 'w') as f:
    json.dump(heart_center_relative_coordinates, f)
