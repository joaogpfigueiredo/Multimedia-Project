import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct,idct

cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
cm_grey = clr.LinearSegmentedColormap.from_list("grey", [(0, 0, 0), (1, 1, 1)], N=256)

# General Functions
def showImage(img, cm, title="Imagem"):
    plt.figure()
    plt.imshow(img, cmap=cm)
    plt.axis('off')
    plt.title(title)
    plt.show()

def showImageDCT(img, cm, title="Imagem"):
    plt.figure()
    plt.imshow(np.log(abs(img)+0.0001), cmap=cm)
    plt.axis('off')
    plt.title(title)
    plt.show()

def channels_to_img(R, G, B):
    nl, nc = R.shape
    img = np.zeros((nl, nc, 3), dtype=np.uint8)

    img[:, :, 0] = R
    img[:, :, 1] = G
    img[:, :, 2] = B

    return img


def showSubMatrix(matrix, i, j, dim):
    nd = matrix.ndim
    
    if nd == 2:
        print(matrix[i:i+dim, j:j+dim])
    elif nd == 3:
        print(matrix[i:i+dim, j:j+dim, 0])

# Ex 4 
def padding(img, block_size = 32):
    height, width, __ = img.shape
    
    pad_height = block_size - (height % block_size) if height % block_size != 0 else 0
    pad_width = block_size - (width % block_size) if width % block_size != 0 else 0
    img_pad = np.pad(img, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')
    
    return img_pad


def remove_padding(image_padded, original_shape):
    height, width, __ = original_shape
    
    original_img = image_padded[:height, :width, :]

    return original_img
    
# Ex 5
def rgb_to_ycbcr(img):
    matrix = np.array([[0.299, 0.587, 0.114], 
                       [-0.168736, -0.331264, 0.5], 
                       [0.5, -0.418688, -0.081312]])
    
    Y = matrix[0][0] * img[:, :, 0] + matrix[0][1] * img[:, :, 1] + matrix[0][2] * img[:, :, 2] 
    Cb = matrix[1][0] * img[:, :, 0] + matrix[1][1] * img[:, :, 1] + matrix[1][2] * img[:, :, 2] + 128
    Cr = matrix[2][0] * img[:, :, 0] + matrix[2][1] * img[:, :, 1] + matrix[2][2] * img[:, :, 2] + 128
    
    return Y, Cb, Cr


def ycbcr_to_rgb(Y , Cb, Cr):
    inv_matrix = np.linalg.inv([[0.299, 0.587, 0.114], 
                                [-0.168736, -0.331264, 0.5], 
                                [0.5, -0.418688, -0.081312]])
    
    R = Y * inv_matrix[0][0] + (Cb - 128) * inv_matrix[0][1] + (Cr - 128) * inv_matrix[0][2]
    G = Y * inv_matrix[1][0] + (Cb - 128) * inv_matrix[1][1] + (Cr - 128) * inv_matrix[1][2]
    B = Y * inv_matrix[2][0] + (Cb - 128) * inv_matrix[2][1] + (Cr - 128) * inv_matrix[2][2]
    
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)

    R = np.round(R).astype(np.uint8)
    G = np.round(G).astype(np.uint8)
    B = np.round(B).astype(np.uint8)
    
    return R, G, B

# Ex 6
def downsampling(Y, Cb, Cr, variant, interpolation):
    if(variant == [4,2,0]):
        if(interpolation == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_LINEAR)
        elif(interpolation == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC)
    elif(variant == [4,2,2]):
        if(interpolation == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_LINEAR)
        elif(interpolation == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_CUBIC)
        
    return Y, Cb_d, Cr_d


def upsampling(Y, Cb, Cr, variant, interpolation):
    if(variant == [4,2,0]):
        if(interpolation == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 2, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 2, interpolation=cv2.INTER_LINEAR)
        elif(interpolation == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 2, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 2, interpolation=cv2.INTER_CUBIC)
    elif(variant == [4,2,2]):
        if(interpolation == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 1, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 1, interpolation=cv2.INTER_LINEAR)
        elif(interpolation == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 1, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 1, interpolation=cv2.INTER_CUBIC)
        
    return Y, Cb_d, Cr_d

def get_dct(X):
    return dct(dct(X,norm="ortho").T,norm="ortho").T

def get_idct(X):
    return idct(idct(X,norm="ortho").T,norm="ortho").T


# Encoder and Decoder
def encoder(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    showImage(R, cm_red, "Canal R")
    showImage(G, cm_green, "Canal G")
    showImage(B, cm_blue, "Canal B")

    image_padded = padding(img)

    Y, Cb, Cr = rgb_to_ycbcr(image_padded)
    
    showImage(Y, cm_grey, "Canal Y")
    showImage(Cb, cm_grey, "Canal Cb")
    showImage(Cr, cm_grey, "Canal Cr")
    
    
    print("\n################ DOWNSAMPLING####################\n")
    
    print("Cb shape before Downsampling: ", Cb.shape)
    print("Cd shape before Downsampling: ", Cr.shape)
    
    print("\nVariant[4:2:2]\n")
    
    ######################LINEAR DOWNSAMPLING 4:2:2 #############################
    
    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, [4,2,2],"linear")
    
    showImage(Y_d, cm_grey, " Y (Downsampling (Linear) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Linear) with [4:2:2)")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (Linear) with [4:2:2])")
    
    ######################CUBIC DOWNSAMPLING 4:2:2 #############################
    
    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, [4,2,2],"cubic")
    
    showImage(Y_d, cm_grey, " Y (Downsampling (Cubic) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Cubic) with [4:2:2])")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (cubic) with [4:2:2])")
    
    print("Cb shape after Downsampling([4:2:2]):", Cb_d.shape)
    print("Cd shape after Downsampling([4:2:2]):", Cr_d.shape)
     
    '''
    print("\nVariant[4:2:0]\n")
    
    ######################LINEAR DOWNSAMPLING 4:2:0 #############################
    
    Y, Cb_d, Cr_d = downsampling(Y, Cb, Cr, [4,2,0], "linear")
    
    showImage(Y, cm_grey, " Y (Downsampling (Linear) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Linear) with [4:2:0)")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (Linear) with [4:2:0])")
    
    ######################CUBIC DOWNSAMPLING 4:2:0 #############################
    
    Y, Cb_d, Cr_d = downsampling(Y, Cb, Cr, [4,2,0], "cubic")
    
    showImage(Y, cm_grey, " Y (Downsampling (Cubic) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Cubic) with [4:2:0])")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (cubic) with [4:2:0])")
    
    print("Cb shape after Downsampling([4:2:2]):", Cb_d.shape)
    print("Cd shape after Downsampling([4:2:2]):", Cr_d.shape)
    
    print("\n################################################")
    
    print("\n######################DCT#######################\n")
    '''
   
    Y_dct = get_dct(Y_d)
    Cb_dct = get_dct(Cb_d)
    Cr_dct = get_dct(Cr_d)
    
    showImageDCT(Y_dct, cm_grey,"DCT IN Y")
    showImageDCT(Cb_dct, cm_grey,"DCT IN CB_d")
    showImageDCT(Cr_dct, cm_grey,"DCT IN Cr_d")

    

    return Y, Cb, Cr, Y_dct, Cb_dct, Cr_dct


def decoder(Y, Cb, Cr, Y_dct, Cb_dct, Cr_dct):
    
    Y_idct = get_idct(Y_dct)
    Cb_idct = get_idct(Cb_dct)
    Cr_idct = get_idct(Cr_dct)
    
    showImageDCT(Y_idct, cm_grey,"IDCT IN Y")
    showImageDCT(Cb_idct, cm_grey,"IDCT IN CB_d")
    showImageDCT(Cr_idct, cm_grey,"IDCT IN Cr_d")
    
    print("\n################ UPSAMPLING####################\n")
    
    print("Cb shape before Upsampling: ",Cb.shape)
    print("Cd shape before Upsampling: ",Cr.shape)
    
    
    print("\nVariant[4:2:2]\n")
    
    ######################LINEAR DOWNSAMPLING 4:2:2 #############################
    
    Y_d, Cb_d, Cr_d = upsampling(Y, Cb, Cr, [4,2,2], "linear")
    
    showImage(Y_d, cm_grey, " Y (Upsampling (Linear) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Linear) with [4:2:2)")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (Linear) with [4:2:2])")
    
    ######################CUBIC DOWNSAMPLING 4:2:2 #############################
    
    Y_d, Cb_d, Cr_d = upsampling(Y, Cb, Cr, [4,2,2], "cubic") 
    
    showImage(Y_d, cm_grey, " Y (Upsampling (Cubic) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Cubic) with [4:2:2])")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (cubic) with [4:2:2])")
    
    print("Cb shape after Upsampling([4:2:2]):", Cb_d.shape)
    print("Cd shape after Upsampling([4:2:2]):", Cr_d.shape)
    
    '''
    print("\nVariant[4:2:0]\n")
    
    ######################LINEAR DOWNSAMPLING 4:2:0 #############################
    
    Y, Cb_d, Cr_d = upsampling(Y, Cb, Cr, [4,2,0], "linear")
    
    showImage(Y, cm_grey, " Y (Upsampling (Linear) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Linear) with [4:2:0)")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (Linear) with [4:2:0])")
    
    ######################CUBIC DOWNSAMPLING 4:2:0 #############################
    
    Y, Cb_d, Cr_d = upsampling(Y, Cb, Cr, [4,2,0], "cubic")
    
    showImage(Y, cm_grey, " Y (Upsampling (Cubic) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Cubic) with [4:2:0])")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (cubic) with [4:2:0])")
    
    print("Cb shape after Upsampling([4:2:0]):", Cb_d.shape)
    print("Cd shape after Upsampling([4:2:0]):", Cr_d.shape)
    '''
    
    print("\n################################################")
    
    

    R, G, B = ycbcr_to_rgb(Y, Cb, Cr)

    img = channels_to_img(R, G, B)

    original_img = remove_padding(img, img.shape)
    
    return original_img

def main():
    filename = "imagens/airport.bmp"
    img = plt.imread(filename)
    showImage(img, None, "Original Image")
    
    # print("Image type:", type(img))
    # print("Image shape:", img.shape)
    
    # print(img[0:8, 0:8, 0])
    # print("Image data type:", img.dtype)
    
    # showSubMatrix(img, 0, 0, 8)
    
    Y, Cb, Cr, Y_dct, Cb_dct, Cr_dct = encoder(img)
    
    imgRec = decoder(Y, Cb, Cr, Y_dct, Cb_dct, Cr_dct)
    showImage(imgRec, None, "Reconstructed Image")
    
if __name__ == "__main__":
    main()