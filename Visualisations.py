### Funkce:
def CreateSlicers(image_3d, manual=False):
    """
    Vytvoří interaktivní slicer pro 3D obraz ovládaný kolečkem myši
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import numpy as np 

    if manual == False:
        image_3d = np.flip(image_3d, axis=0)

    # Vytvoření figure s třemi subploty
    fig = plt.figure(figsize=(15, 6))
    
    # Axiální pohled
    ax1 = plt.subplot(131)
    sl1 = image_3d[image_3d.shape[0]//2,:,:]
    img1 = ax1.imshow(sl1, cmap='gray')
    ax1.set_title('Axiální rovina (XY)\nZ: {}/{}'.format(image_3d.shape[0]//2, image_3d.shape[0]-1))
    
    # Sagitální pohled
    ax2 = plt.subplot(132)
    sl2 = image_3d[:,:,image_3d.shape[2]//2]
    img2 = ax2.imshow(sl2, cmap='gray')
    ax2.set_title('Sagitální rovina (YZ)\nX: {}/{}'.format(image_3d.shape[2]//2, image_3d.shape[2]-1))
    
    # Koronální pohled
    ax3 = plt.subplot(133)
    sl3 = image_3d[:,image_3d.shape[1]//2,:]
    img3 = ax3.imshow(sl3, cmap='gray')
    ax3.set_title('Koronální rovina (XZ)\nY: {}/{}'.format(image_3d.shape[1]//2, image_3d.shape[1]-1))
    
    # Inicializace indexů řezů
    z_index = image_3d.shape[0]//2
    x_index = image_3d.shape[2]//2
    y_index = image_3d.shape[1]//2
    
    def on_scroll(event):
        nonlocal z_index, x_index, y_index
        
        # Zjištění, který subplot je aktivní
        if event.inaxes == ax1:  # Axiální pohled
            z_index = (z_index + int(event.step)) % image_3d.shape[0]
            img1.set_data(image_3d[z_index,:,:])
            ax1.set_title('Axiální rovina (XY)\nZ: {}/{}'.format(z_index, image_3d.shape[0]-1))
            
        elif event.inaxes == ax2:  # Sagitální pohled
            x_index = (x_index + int(event.step)) % image_3d.shape[2]
            img2.set_data(image_3d[:,:,x_index])
            ax2.set_title('Sagitální rovina (YZ)\nX: {}/{}'.format(x_index, image_3d.shape[2]-1))
            
        elif event.inaxes == ax3:  # Koronální pohled
            y_index = (y_index + int(event.step)) % image_3d.shape[1]
            img3.set_data(image_3d[:,y_index,:])
            ax3.set_title('Koronální rovina (XZ)\nY: {}/{}'.format(y_index, image_3d.shape[1]-1))
        
        fig.canvas.draw_idle()
    
    # Připojení události scrollování
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    plt.tight_layout()
    return fig


def create_interactive_coordinate_picker(image):
    """
    Vytvoří interaktivní okno pro výběr souřadnic z různých pohledů
    s ovládáním pomocí scrollovacího tlačítka myši
    
    Args:
        image: SimpleITK obraz (již opravený, bez potřeby flippingu)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import numpy as np
    import SimpleITK as sitk

    # Převedeme SimpleITK obraz na numpy array
    display_array = sitk.GetArrayFromImage(image)
    
    # Vytvoříme figure s třemi subploty
    fig = plt.figure(figsize=(15, 5))
    
    # Axial view (xy)
    ax1 = fig.add_subplot(131)
    axial_slice = display_array[display_array.shape[0]//2, :, :]
    axial_img = ax1.imshow(axial_slice, cmap='gray')
    ax1.set_title('Axial view (XY)\nZ: {}/{}'.format(display_array.shape[0]//2, display_array.shape[0]-1))
    
    # Sagittal view (yz)
    ax2 = fig.add_subplot(132)
    sagittal_slice = display_array[:, :, display_array.shape[2]//2]
    sagittal_img = ax2.imshow(sagittal_slice, cmap='gray')
    ax2.set_title('Sagittal view (YZ)\nX: {}/{}'.format(display_array.shape[2]//2, display_array.shape[2]-1))
    
    # Coronal view (xz)
    ax3 = fig.add_subplot(133)
    coronal_slice = display_array[:, display_array.shape[1]//2, :]
    coronal_img = ax3.imshow(coronal_slice, cmap='gray')
    ax3.set_title('Coronal view (XZ)\nY: {}/{}'.format(display_array.shape[1]//2, display_array.shape[1]-1))
    
    # Inicializace indexů řezů
    z_index = display_array.shape[0]//2
    x_index = display_array.shape[2]//2
    y_index = display_array.shape[1]//2
    
    # Globální proměnné pro ukládání souřadnic
    coordinates = {'x': None, 'y': None, 'z': None}
    
    def on_scroll(event):
        nonlocal z_index, x_index, y_index
        
        # Zjištění, který subplot je aktivní
        if event.inaxes == ax1:  # Axial view
            z_index = (z_index + int(event.step)) % display_array.shape[0]
            axial_img.set_data(display_array[z_index,:,:])
            ax1.set_title('Axial view (XY)\nZ: {}/{}'.format(z_index, display_array.shape[0]-1))
            
        elif event.inaxes == ax2:  # Sagittal view
            x_index = (x_index + int(event.step)) % display_array.shape[2]
            sagittal_img.set_data(display_array[:,:,x_index])
            ax2.set_title('Sagittal view (YZ)\nX: {}/{}'.format(x_index, display_array.shape[2]-1))
            
        elif event.inaxes == ax3:  # Coronal view
            y_index = (y_index + int(event.step)) % display_array.shape[1]
            coronal_img.set_data(display_array[:,y_index,:])
            ax3.set_title('Coronal view (XZ)\nY: {}/{}'.format(y_index, display_array.shape[1]-1))
        
        fig.canvas.draw_idle()
    
    def on_axial_click(event):
        if event.inaxes == ax1:
            x = int(event.xdata)
            y = int(event.ydata)
            coordinates['x'] = x
            coordinates['y'] = y
            coordinates['z'] = z_index  # Použijeme aktuální hodnotu z_index
#            print(f"X,Y,Z coordinates set to: ({coordinates['x']}, {coordinates['y']}, {coordinates['z']})")
    
    def on_sagittal_click(event):
        if event.inaxes == ax2:
            z = int(event.ydata)
            coordinates['z'] = z
#            print(f"Z coordinate set to: {coordinates['z']}")
    
    def on_coronal_click(event):
        if event.inaxes == ax3:
            z = int(event.ydata)
            coordinates['z'] = z
#            print(f"Z coordinate set to: {coordinates['z']}")
    
    def save_coordinates(event):
        if all(v is not None for v in coordinates.values()):
            print(f"Final coordinates: ({coordinates['x']}, {coordinates['y']}, {coordinates['z']})")
            plt.close()
        else:
            print("Please set all coordinates first!")
    
    # Přidáme tlačítko pro uložení
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
    button = Button(ax_button, 'Save')
    button.on_clicked(save_coordinates)
    
    # Přidáme event handlery
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_axial_click)
    fig.canvas.mpl_connect('button_press_event', on_sagittal_click)
    fig.canvas.mpl_connect('button_press_event', on_coronal_click)
    
    plt.tight_layout()
    plt.show()
    
    return coordinates
