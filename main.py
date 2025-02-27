import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
from scipy.fftpack import dct,idct

cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
cm_grey = clr.LinearSegmentedColormap.from_list("grey", [(0, 0, 0), (1, 1, 1)], N=256)

ycbcr_matrix = np.array([[0.299, 0.587, 0.114], 
                       [-0.168736, -0.331264, 0.5], 
                       [0.5, -0.418688, -0.081312]])

quantization_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])

quantization_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                [18, 21, 26, 66, 99, 99, 99, 99],
                                [24, 26, 56, 99, 99, 99, 99, 99],
                                [47, 66, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99],
                                [99, 99, 99, 99, 99, 99, 99, 99]])

# General Functions
def showImage(img, cm, title="Imagem"):
    plt.figure()
    plt.imshow(img, cmap=cm)
    plt.axis('off')
    plt.title(title)
    plt.show()

def showImageDCT(img, cm, title="Imagem"):
    plt.figure()
    plt.imshow(np.log(np.abs(img)+0.0001), cmap=cm)
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
        print(np.round(matrix[i:i+dim, j:j+dim],3))
    elif nd == 3:
        print(np.round(matrix[i:i+dim, j:j+dim, 0],3))
        
def showSubMatrix_nr(matrix, i, j, dim):
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
    
    Y = ycbcr_matrix[0][0] * img[:, :, 0] + ycbcr_matrix[0][1] * img[:, :, 1] + ycbcr_matrix[0][2] * img[:, :, 2] 
    Cb = ycbcr_matrix[1][0] * img[:, :, 0] + ycbcr_matrix[1][1] * img[:, :, 1] + ycbcr_matrix[1][2] * img[:, :, 2] + 128
    Cr = ycbcr_matrix[2][0] * img[:, :, 0] + ycbcr_matrix[2][1] * img[:, :, 1] + ycbcr_matrix[2][2] * img[:, :, 2] + 128
    
    return Y, Cb, Cr

def ycbcr_to_rgb(Y , Cb, Cr):
    inv_matrix = np.linalg.inv(ycbcr_matrix)
    
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


# Ex 7
def get_dct(X):
    return dct(dct(X,norm="ortho").T,norm="ortho").T

def get_idct(X):
    return idct(idct(X,norm="ortho").T,norm="ortho").T

def dct_by_block(channel, blockSize):
    rows = channel.shape[0]
    columns = channel.shape[1]
    final_dct = np.zeros(channel.shape)
    
    for i in range(0,rows,blockSize):
        for j in range(0,columns,blockSize):
            portion = channel[i:i+blockSize, j:j+blockSize]
            final_dct[i:i+blockSize, j:j+blockSize] = get_dct(portion)
            
    return final_dct

def idct_by_block(channel, blockSize):
    rows = channel.shape[0]
    columns = channel.shape[1]
    final_idct = np.zeros(channel.shape)
    
    for i in range(0, rows, blockSize):
        for j in range(0, columns, blockSize):
            portion = channel[i:i + blockSize, j:j+blockSize]
            final_idct[i:i + blockSize, j:j+blockSize] = get_idct(portion)
            
    return final_idct


# Ex 8
def qualityCalc(quality):
    matrix_ones = np.ones((8, 8))
    
    if quality >= 50:
        scaleFactor = (100 - quality) / 50
    else:
        scaleFactor = 50 / quality

    if scaleFactor == 0:
        qualityQ_Y = matrix_ones
        qualityQ_CbCr = matrix_ones
    else:
        qualityQ_Y = quantization_y * scaleFactor
        qualityQ_CbCr = quantization_cbcr * scaleFactor


    qualityQ_Y = np.clip(qualityQ_Y, 1, 255).astype(np.uint8)
    qualityQ_CbCr = np.clip(qualityQ_CbCr, 1, 255).astype(np.uint8)

    return qualityQ_Y, qualityQ_CbCr

def quantization(Y, Cb, Cr, quality, blocks):
    
    qualityQ_Y, qualityQ_CbCr = qualityCalc(quality)

    length = Y.shape
    for i in range(0, length[0], blocks):
        for j in range(0, length[1], blocks):
            slice_Y = Y[i:i + blocks, j:j + blocks]
            Y[i:i + blocks, j:j + blocks] = slice_Y / qualityQ_Y                          

    length = Cb.shape
    for i in range(0, length[0], blocks):
        for j in range(0, length[1], blocks):
            slice_Cb = Cb[i:i + blocks, j:j + blocks]
            Cb[i:i + blocks, j:j + blocks] = slice_Cb / qualityQ_CbCr

            slice_Cr = Cr[i:i + blocks, j:j + blocks]
            Cr[i:i + blocks, j:j + blocks] = slice_Cr / qualityQ_CbCr

    Y = np.round(Y).astype(int)
    Cb = np.round(Cb).astype(int)
    Cr = np.round(Cr).astype(int)

    showImage(np.log(abs(Y) + 0.0001), cm_grey, 'Y Quantized')
    showImage(np.log(abs(Cb) + 0.0001), cm_grey, 'Cb Quantized')
    showImage(np.log(abs(Cr) + 0.0001), cm_grey, 'Cr Quantized')

    dict_Q = {'Yb_Q': Y, 'Cbb_Q': Cb, 'Crb_Q': Cr}

    return dict_Q
    
def iquantization(dict_Q, quality, blocks):
    
    Y, Cb, Cr = dict_Q.values()
    
    qualityQ_Y, qualityQ_CbCr = qualityCalc(quality)

    length = Y.shape
    for i in range(0, length[0], blocks):
        for j in range(0, length[1], blocks):
            slice = Y[i:i + blocks, j:j + blocks]
            Y[i:i + blocks, j:j + blocks] = slice * qualityQ_Y

            
    length = Cb.shape
    for i in range(0, length[0], blocks):
        for j in range(0, length[1], blocks):
            slice = Cb[i:i + blocks, j:j + blocks]
            Cb[i:i + blocks, j:j + blocks] = slice * qualityQ_CbCr

            slice = Cr[i:i + blocks, j:j + blocks]
            Cr[i:i + blocks, j:j + blocks] = slice * qualityQ_CbCr

    Y  = Y.astype(float)
    Cb = Cb.astype(float)
    Cr = Cr.astype(float)

    showImage(np.log(abs(Y) + 0.0001), cm_grey, 'Y Iquantization')
    showImage(np.log(abs(Cb) + 0.0001), cm_grey, 'Cb Iquantization')
    showImage(np.log(abs(Cr) + 0.0001), cm_grey, 'Cr Iquantization')
    
    dct_dict = {'Y_dct': Y, 'Cb_dct': Cb, 'Cr_dct': Cr}

    return dct_dict

# Encoder and Decoder
def encoder(img, mode, factor, quality):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    showImage(R, cm_red, "Canal R")
    showImage(G, cm_green, "Canal G")
    showImage(B, cm_blue, "Canal B")
    
    print("Matriz R")
    showSubMatrix(R, 8, 8, 8)

    image_padded = padding(img)

    Y, Cb, Cr = rgb_to_ycbcr(image_padded)
    
    showImage(Y, cm_grey, "Canal Y")
    showImage(Cb, cm_grey, "Canal Cb")
    showImage(Cr, cm_grey, "Canal Cr")
    
    print("Matriz Y")
    showSubMatrix(Y, 8, 8, 8)
    
    print("\nMatriz Cb")
    showSubMatrix(Cb, 8, 8, 8)
    
    
    print("\n################ DOWNSAMPLING####################\n")
    
    print("Cb shape before Downsampling: ", Cb.shape)
    print("Cd shape before Downsampling: ", Cr.shape)
    
    print(f"\nVariant{factor}\n")
    
    ###################### DOWNSAMPLING #############################
    
    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, factor,mode)
    
    showImage(Y_d, cm_grey, f"Y (Downsampling ({mode}) with {factor})")
    showImage(Cb_d, cm_grey, f"Cb (Downsampling ({mode}) with {factor})")
    showImage(Cr_d, cm_grey, f"Cr (Downsampling ({mode}) with {factor})")
    
    print("Matriz Cb")
    showSubMatrix(Cb,8,8,8)

    
    print(f"Cb shape after Downsampling({factor}):", Cb_d.shape)
    print(f"Cb shape after Downsampling({factor}):", Cr_d.shape)
   
   ###################### EX 7.1 #############################
    Y_dct = get_dct(Y_d)
    Cb_dct = get_dct(Cb_d)
    Cr_dct = get_dct(Cr_d)
    
    showImageDCT(Y_dct, cm_grey,"DCT IN Y")
    showImageDCT(Cb_dct, cm_grey,"DCT IN Cb_d")
    showImageDCT(Cr_dct, cm_grey,"DCT IN Cr_d")
    
    ###################### EX 7.2 #############################
    Y_dct_block8 = dct_by_block(Y_d, 8)
    Cb_dct_block8 = dct_by_block(Cb_d, 8)
    Cr_dct_block8 = dct_by_block(Cr_d, 8)
    
    showImageDCT(Y_dct_block8, cm_grey,"DCT8 IN Y")
    showImageDCT(Cb_dct_block8, cm_grey,"DCT8 IN Cb_d")
    showImageDCT(Cr_dct_block8, cm_grey,"DCT8 IN Cr_d")
    
    print("Matriz Yb_DCT8X8")
    showSubMatrix_nr(Y_dct_block8, 8, 8, 8)
    
    '''
    ###################### EX 7.3 #############################
    Y_dct_block64 = dct_by_block(Y_d, 64)
    Cb_dct_block64 = dct_by_block(Cb_d, 64)
    Cr_dct_block64 = dct_by_block(Cr_d, 64)
    
    showImageDCT(Y_dct_block64, cm_grey,"DCT64 IN Y")
    showImageDCT(Cb_dct_block64, cm_grey,"DCT64 IN Cb_d")
    showImageDCT(Cr_dct_block64, cm_grey,"DCT64 IN Cr_d")
    '''
    
    dct8_dict = {'Y': Y_dct_block8, 'Cb': Cb_dct_block8, 'Cr': Cr_dct_block8}
    
    dict_Q = quantization(dct8_dict['Y'], dct8_dict['Cb'], dct8_dict['Cr'], quality, 8)
    showSubMatrix(dict_Q['Yb_Q'], 8, 8, 8)

    return dict_Q


def decoder(original_img, dict_Q, mode, factor, quality):
    
    ###################### EX 8.2 #############################
    dct8_dict = iquantization(dict_Q, quality, 8)
    Y_dct8, Cb_dct8, Cr_dct8 = dct8_dict.values()
    showSubMatrix(Y_dct8, 8, 8, 8)
    
    '''
    ###################### EX 7.1 #############################
    Y_d = get_idct(Y_dct)
    Cb_d = get_idct(Cb_dct)
    Cr_d = get_idct(Cr_dct)
    
    showImageDCT(Y_d, cm_grey,"IDCT IN Y")
    showImageDCT(Cb_d, cm_grey,"IDCT IN Cb_d")
    showImageDCT(Cr_d, cm_grey,"IDCT IN Cr_d")
    '''
    
    ###################### EX 7.2 #############################
    Y_d8 = idct_by_block(Y_dct8, 8)
    Cb_d8 = idct_by_block(Cb_dct8, 8)
    Cr_d8 = idct_by_block(Cr_dct8, 8)
    
    showImageDCT(Y_d8, cm_grey,"IDCT8 IN Y")
    showImageDCT(Cb_d8, cm_grey,"IDCT8 IN Cb_d")
    showImageDCT(Cr_d8, cm_grey,"IDCT8 IN Cr_d")
    
    '''
    ###################### EX 7.3 #############################
    Y_d64 = idct_by_block(Y_dct64, 64)
    Cb_d64 = idct_by_block(Cb_dct64, 64)
    Cr_d64 = idct_by_block(Cr_dct64, 64)
    
    showImageDCT(Y_d64, cm_grey,"IDCT64 IN Y")
    showImageDCT(Cb_d64, cm_grey,"IDCT64 IN Cb_d")
    showImageDCT(Cr_d64, cm_grey,"IDCT64 IN Cr_d")
    '''
    
    print("\n################ UPSAMPLING####################\n")
    
    print("Cb shape before Upsampling: ",Cb_d8.shape)
    print("Cd shape before Upsampling: ",Cr_d8.shape)
    
    
    print(f"\nVariant{factor}\n")
    
    ######################UPSAMPLING #############################
    
    Y, Cb, Cr = upsampling(Y_d8, Cb_d8, Cr_d8, factor,mode) 
    
    showImage(Y, cm_grey, f" Y (Upsampling ({mode}) with {factor})")
    showImage(Cb, cm_grey, f"Cb (Upsampling ({mode}) with {factor})")
    showImage(Cr, cm_grey, f"Cr (Upsampling ({mode}) with {factor})")
    
    print(f"Cb shape after Upsampling({factor}):", Cb.shape)
    print(f"Cd shape after Upsampling({factor}):", Cr.shape)
    
    print("\n################################################")
    

    R, G, B = ycbcr_to_rgb(Y, Cb, Cr)

    img = channels_to_img(R, G, B)

    original_img = remove_padding(img, original_img.shape)
    
    '''
    print("Imagem recuperada")
    showSubMatrix(original_img,8,8,8)
    '''
    
    return original_img

def main():
    filename = "imagens/airport.bmp"
    img = plt.imread(filename)
    showImage(img, None, "Original Image")
    
    '''
    print("Imagem original")
    showSubMatrix(img,8,8,8)
    '''
    
    # print("Image type:", type(img))
    # print("Image shape:", img.shape)
    
    # print(img[0:8, 0:8, 0])
    # print("Image data type:", img.dtype)
    
    # showSubMatrix(img, 0, 0, 8)
    
    mode = "linear"
    #mode = "cubic"
    
    factor = [4, 2, 2]
    #factor = [4,2,0]
    
    dict_Q = encoder(img, mode, factor, 75)
    
    imgRec = decoder(img, dict_Q, mode, factor, 75)
    showImage(imgRec, None, "Reconstructed Image")
    
if __name__ == "__main__":
    main()