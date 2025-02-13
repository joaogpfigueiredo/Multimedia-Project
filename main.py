import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2

cm_red = clr.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
cm_grey = clr.LinearSegmentedColormap.from_list("grey", [(0, 0, 0), (1, 1, 1)], N=256)

def showImage(img, cm, title="Imagem"):
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

def pad_channel(channel, block_size=32):
    nl, nc = channel.shape
    
    pad_nl = (block_size - nl % block_size) % block_size
    pad_nc = (block_size - nc % block_size) % block_size
    padded_channel = np.pad(channel, ((0, pad_nl), (0, pad_nc)), mode='edge')
    
    return padded_channel, nl, nc

def remove_padding(channel, original_nl, original_nc):
    return channel[:original_nl, :original_nc]

def rgb_to_ycbcr(img):
    
    Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    Cb = 128 - 0.168736 * img[:, :, 0] - 0.331264 * img[:, :, 1] + 0.5 * img[:, :, 2]
    Cr = 128 + 0.5 * img[:, :, 0] - 0.418688 * img[:, :, 1] - 0.081312 * img[:, :, 2]
    
    return np.stack((Y, Cb, Cr), axis=-1)

def ycbcr_to_rgb(img):
    
    R = img[:, :, 0] + 1.402 * (img[:, :, 2] - 128)
    G = img[:, :, 0] - 0.344136 * (img[:, :, 1] - 128) - 0.714136 * (img[:, :, 2] - 128)
    B = img[:, :, 0] + 1.772 * (img[:, :, 1] - 128)
    rgb_img = np.stack((R, G, B), axis=-1)
    
    return np.clip(rgb_img, 0, 255).astype(np.uint8)

def downsampling(Y,Cb,Cr,modo,interpolacao):
    if(modo == [4,2,0]):
        if(interpolacao == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_LINEAR)
        elif(interpolacao == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 0.5, interpolation=cv2.INTER_CUBIC)
    elif(modo == [4,2,2]):
        if(interpolacao == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_LINEAR)
        elif(interpolacao == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 0.5,fy = 1, interpolation=cv2.INTER_CUBIC)
        
    return Y, Cb_d,Cr_d


def upsampling(Y,Cb,Cr,modo,interpolacao):
    if(modo == [4,2,0]):
        if(interpolacao == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 2, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 2, interpolation=cv2.INTER_LINEAR)
        elif(interpolacao == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 2, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 2, interpolation=cv2.INTER_CUBIC)
    elif(modo == [4,2,2]):
        if(interpolacao == "linear"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 1, interpolation=cv2.INTER_LINEAR)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 1, interpolation=cv2.INTER_LINEAR)
        elif(interpolacao == "cubic"):
            Cb_d = cv2.resize(Cb,None,fx = 2,fy = 1, interpolation=cv2.INTER_CUBIC)
            Cr_d = cv2.resize(Cr,None,fx = 2,fy = 1, interpolation=cv2.INTER_CUBIC)
        
    return Y, Cb_d,Cr_d


def encoder(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    showImage(R, cm_red, "Canal R")
    showImage(G, cm_green, "Canal G")
    showImage(B, cm_blue, "Canal B")

    ycbcr_img = rgb_to_ycbcr(img)
    Y = ycbcr_img[:, :, 0]
    Cb = ycbcr_img[:, :, 1]
    Cr = ycbcr_img[:, :, 2]
    
    showImage(Y, cm_grey, "Canal Y")
    showImage(Cb, cm_grey, "Canal Cb")
    showImage(Cr, cm_grey, "Canal Cr")

    Y_padded, Y_nl, Y_nc = pad_channel(Y)
    Cb_padded, Cb_nl, Cb_nc = pad_channel(Cb)
    Cr_padded, Cr_nl, Cr_nc = pad_channel(Cr)

    return (Y_padded, Y_nl, Y_nc), (Cb_padded, Cb_nl, Cb_nc), (Cr_padded, Cr_nl, Cr_nc)


def decoder(Y_info, Cb_info, Cr_info):
    Y_padded, Y_nl, Y_nc = Y_info
    Cb_padded, Cb_nl, Cb_nc = Cb_info
    Cr_padded, Cr_nl, Cr_nc = Cr_info
    
    Y = remove_padding(Y_padded, Y_nl, Y_nc)
    Cb = remove_padding(Cb_padded, Cb_nl, Cb_nc)
    Cr = remove_padding(Cr_padded, Cr_nl, Cr_nc)
    
    ycbcr_img = np.stack((Y, Cb, Cr), axis=-1)
    rgb_img = ycbcr_to_rgb(ycbcr_img)
    
    return rgb_img

def main():
    filename = "imagens/airport.bmp"
    img = plt.imread(filename)
    showImage(img, None, "Original Image")
    
    print("Image type:", type(img))
    print("Image shape:", img.shape)
    
    print(img[0:8, 0:8, 0])
    print("Image data type:", img.dtype)
    
    showSubMatrix(img, 0, 0, 8)
    
    Y_info, Cb_info, Cr_info = encoder(img)
    
    imgRec = decoder(Y_info, Cb_info, Cr_info)
    showImage(imgRec, None, "Reconstructed Image")
    
    ycbcr_img = rgb_to_ycbcr(img)
    Y = ycbcr_img[:, :, 0]
    Cb = ycbcr_img[:, :, 1]
    Cr = ycbcr_img[:, :, 2]
    
    print("\n################ DOWNSAMPLING####################\n")
    print("Cb shape before Downsampling:",Cb.shape)
    print("Cd shape before Downsampling:",Cr.shape)
    
    print("\n   Variante[4:2:2]\n")
    
    ######################LINEAR DOWNSAMPLING#############################
    
    Y,Cb_d,Cr_d = downsampling(Y, Cb, Cr, [4,2,2],"linear")
    
    showImage(Y, cm_grey, " Y (Downsampling (Linear) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Linear) with [4:2:2)")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (Linear) with [4:2:2])")
    
    ######################CUBIC DOWNSAMPLING#############################
    
    showImage(Y, cm_grey, " Y (Downsampling (Cubic) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Cubic) with [4:2:2])")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (cubic) with [4:2:2])")
    
    print("Cb shape after Downsampling([4:2:2]):",Cb_d.shape)
    print("Cd shape after Downsampling([4:2:2]):",Cr_d.shape)
    
    print("\n   Variante[4:2:0]\n")
    
    ######################LINEAR DOWNSAMPLING#############################
    
    Y,Cb_d,Cr_d = downsampling(Y, Cb, Cr, [4,2,0],"linear")
    
    showImage(Y, cm_grey, " Y (Downsampling (Linear) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Linear) with [4:2:0)")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (Linear) with [4:2:0])")
    
    ######################CUBIC DOWNSAMPLING#############################
    
    showImage(Y, cm_grey, " Y (Downsampling (Cubic) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Downsampling (Cubic) with [4:2:0])")
    showImage(Cr_d, cm_grey, "Cr (Downsampling (cubic) with [4:2:0])")
    
    print("Cb shape after Downsampling([4:2:2]):",Cb_d.shape)
    print("Cd shape after Downsampling([4:2:2]):",Cr_d.shape)
    
    print("\n################################################")
    
    print("\n################ UPSAMPLING####################\n")
    print("Cb shape before Downsampling:",Cb.shape)
    print("Cd shape before Downsampling:",Cr.shape)
    
    print("\n   Variante[4:2:2]\n")
    
    ######################LINEAR DOWNSAMPLING#############################
    
    Y,Cb_d,Cr_d = upsampling(Y, Cb, Cr, [4,2,2],"linear")
    
    showImage(Y, cm_grey, " Y (Upsampling (Linear) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Linear) with [4:2:2)")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (Linear) with [4:2:2])")
    
    ######################CUBIC DOWNSAMPLING#############################
    
    showImage(Y, cm_grey, " Y (Upsampling (Cubic) with [4:2:2])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Cubic) with [4:2:2])")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (cubic) with [4:2:2])")
    
    print("Cb shape after Upsampling([4:2:2]):",Cb_d.shape)
    print("Cd shape after Upsampling([4:2:2]):",Cr_d.shape)
    
    print("\n   Variante[4:2:0]\n")
    
    ######################LINEAR DOWNSAMPLING#############################
    
    Y,Cb_d,Cr_d = upsampling(Y, Cb, Cr, [4,2,0],"linear")
    
    showImage(Y, cm_grey, " Y (Upsampling (Linear) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Linear) with [4:2:0)")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (Linear) with [4:2:0])")
    
    ######################CUBIC DOWNSAMPLING#############################
    
    showImage(Y, cm_grey, " Y (Upsampling (Cubic) with [4:2:0])")
    showImage(Cb_d, cm_grey, "Cb (Upsampling (Cubic) with [4:2:0])")
    showImage(Cr_d, cm_grey, "Cr (Upsampling (cubic) with [4:2:0])")
    
    print("Cb shape after Upsampling([4:2:2]):",Cb_d.shape)
    print("Cd shape after Upsampling([4:2:2]):",Cr_d.shape)
    
    print("\n################################################")
    
    
    
if __name__ == "__main__":
    main()

