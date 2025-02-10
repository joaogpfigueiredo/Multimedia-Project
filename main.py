import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
cm_grey = clr.LinearSegmentedColormap.from_list("grey", [(0, 0, 0), (1, 1, 1)], N=256)

def showImage(img, cm, title):
    plt.figure()
    plt.imshow(img, cmap=cm)
    plt.axis('off')
    plt.title(title)
    plt.show()

def showSubMatrix(matrix, i, j, dim):
    nd = matrix.ndim
    
    if nd == 2:
        print(matrix[i:i+dim, j:j+dim])
    elif nd == 3:
        print(matrix[i:i+dim, j:j+dim, 0])

def encoder(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    showImage(R, cm_red, "Red Channel")
    showImage(G, cm_green, "Green Channel")
    showImage(B, cm_blue, "Blue Channel")
    
    return R, G, B

def decoder(R, G, B):
    nl , nc = R.shape
    imgRec = np.zeros((nl, nc, 3), dtype = np.uint8)
    imgRec[:, :, 0] = R
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B
    
    return imgRec

def main():
    filename = "imagens/airport.bmp"
    img = plt.imread(filename)
    showImage(img, None, "Original Image")
    
    print("Image type:", type(img))
    print("Image shape:", img.shape)
    
    print(img[0:8, 0:8, 0])
    print("Image data type:", img.dtype)
    
    showSubMatrix(img, 0, 0, 8)
    
    R, G, B = encoder(img)
    
    imgRec = decoder(R, G, B)
    showImage(imgRec, None, "Reconstructed Image")
    
if __name__ == "__main__":
    main()

