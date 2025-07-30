import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

def redimensionar_imagem(img, max_largura=800):
    altura, largura = img.shape[:2]
    if largura > max_largura:
        escala = max_largura / largura
        nova_largura = int(largura * escala)
        nova_altura = int(altura * escala)
        img_redimensionada = cv2.resize(img, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
        return img_redimensionada
    return img


def aplicar_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe

def converter_para_cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarizar_imagem(img, threshold=128):
    _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return img_bin

def detectar_bordas_log(img_cinza):
    blur = cv2.GaussianBlur(img_cinza, (3, 3), 0)
    log_edges = cv2.Laplacian(blur, cv2.CV_64F)
    log_abs = np.uint8(np.absolute(log_edges))
    return log_abs


def colocar_titulo(imagem, texto):
    imagem_rotulada = imagem.copy()
    cv2.putText(imagem_rotulada, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return imagem_rotulada


def histogramaEqualizacao(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(imagem_cinza.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    plt.figure()
    plt.subplot(231)
    plt.imshow(imagem_cinza, cmap='gray')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdf_normalized, color='b')
    plt.xlabel('Intensidade')
    plt.ylabel('Número de Pixels')
    
  
    
    # Aplicar CLAHE
    claheObj = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    claheImg = claheObj.apply(imagem_cinza)
    
    claheHist, _ = np.histogram(claheImg.flatten(), 256, [0, 256])
    claheCdf = claheHist.cumsum()
    claheCdf_normalized = claheCdf * claheHist.max() / claheCdf.max()
    plt.subplot(233)
    plt.imshow(claheImg, cmap='gray')
    plt.subplot(236)
    plt.plot(claheHist)
    plt.plot(claheCdf_normalized, color='b')
    plt.xlabel('Intensidade')
    plt.ylabel('Número de Pixels')
    plt.show()





def menu(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Exibe o histograma em uma janela separada usando matplotlib
    histogramaEqualizacao(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    # Etapa 1: Redimensionar
    #img_redim = redimensionar_imagem(img)
    
    # Etapa 2: Aplicar CLAHE
    #img_clahe = aplicar_clahe(img_redim)
    
    # Etapa 3: Converter para escala de cinza
    #img_cinza = converter_para_cinza(img_clahe)
    
    # Etapa 4: Binarizar imagem
    #img_bin = binarizar_imagem(img_cinza)
    
    # Etapa 5: Detectar bordas usando Laplacian of Gaussian
    #img_bordas = detectar_bordas_log(img_cinza)
    
    
    # Exibir imagens com títulos
    #img1 = colocar_titulo(img_redim, "Redimensionada")
    #img2 = colocar_titulo(img_clahe, "CLAHE")
    #img3 = colocar_titulo(cv2.cvtColor(img_cinza, cv2.COLOR_GRAY2BGR), "Escala de Cinza")
    #img4 = colocar_titulo(cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR), "Binarizada")
    #img5 = colocar_titulo(cv2.cvtColor(img_bordas, cv2.COLOR_GRAY2BGR), "Bordas (LoG)")
    
    # Concatenar imagens horizontalmente
    #resultado = cv2.hconcat([img1, img2, img3, img5])
    
    # Exibir resultado em janela única
    #cv2.imshow("Preprocessamento", resultado)
    
menu("traffic.png")
