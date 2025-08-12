import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO E UTILITÁRIOS 
# =============================================================================

# Função para desenhar o gráfico da assinatura da forma em um widget da interface.
# Pré-condição: 'assinatura' é um vetor NumPy (ou None), 'label_widget' é um widget Tkinter válido.
# Pós-condição: O 'label_widget' é limpo e passa a exibir um gráfico Matplotlib da 'assinatura'.
def plotar_assinatura_em_label(assinatura, label_widget, titulo):
    fig = Figure(figsize=(3, 2.5), dpi=100)
    ax = fig.add_subplot(111)

    if assinatura is not None:
        ax.plot(assinatura)
        ax.set_ylim(0, 1.1)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Ponto do Contorno", fontsize=8)
        ax.set_ylabel("Dist. Normalizada", fontsize=8)
    else:
        ax.set_title("Assinatura Inválida", fontsize=10)
    fig.tight_layout()

    for widget in label_widget.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=label_widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Função para exibir uma imagem no formato OpenCV em um widget da interface.
# Pré-condição: 'img_cv' é uma imagem válida no formato NumPy/OpenCV, 'label' é um widget Tkinter Label.
# Pós-condição: O widget 'label' passa a exibir a imagem 'img_cv'.
def exibir_imagem_no_label(img_cv, label):
    img_bgr = garantir_bgr(img_cv)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    for widget in label.winfo_children():
        widget.destroy()
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk

# Função para garantir que uma imagem tenha 3 canais de cor (BGR).
# Pré-condição: 'img' é uma imagem válida no formato NumPy/OpenCV.
# Pós-condição: Retorna a imagem no formato BGR. Se a entrada já era BGR, retorna sem modificação.
def garantir_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO DE IMAGEM 
# =============================================================================

# Função para redimensionar uma imagem se sua maior dimensão exceder um limite.
# Pré-condição: 'img' é uma imagem válida no formato NumPy/OpenCV, 'max_dimensao' é um inteiro.
# Pós-condição: Retorna a imagem redimensionada proporcionalmente se necessário; caso contrário, retorna a imagem original.
def redimensionar_imagem(img, max_dimensao=350):
    if max(img.shape) > max_dimensao:
        fator_escala = max_dimensao / max(img.shape)
        img = cv2.resize(img, (0, 0), fx=fator_escala, fy=fator_escala)
    return img

# Função para aplicar a equalização de histograma adaptativa com contraste limitado (CLAHE).
# Pré-condição: 'img_cinza' é uma imagem válida em tons de cinza.
# Pós-condição: Retorna uma nova imagem em tons de cinza com o contraste melhorado.
def aplicar_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

# Função para converter uma imagem colorida para tons de cinza.
# Pré-condição: 'img_colorida' é uma imagem válida no formato BGR (3 canais).
# Pós-condição: Retorna a imagem convertida para tons de cinza (canal único).
def converter_para_cinza(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Função para detectar as bordas em uma imagem usando o algoritmo Canny.
# Pré-condição: 'img_cinza' é uma imagem válida em tons de cinza; 'limite_inferior' e 'limite_superior' são os limiares do Canny.
# Pós-condição: Retorna uma imagem binária (preto e branco) contendo as bordas detectadas.
def detectar_bordas(img_cinza, limite_inferior=100, limite_superior=120):
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)# A suavização com GaussianBlur é um passo recomendado antes do Canny
    img_canny = cv2.Canny(img_blur, limite_inferior, limite_superior)# Aplica o detector Canny
    return img_canny

# Função para encontrar contornos em uma imagem de bordas e filtrar os que são muito pequenos.
# Pré-condição: 'img_bordas' é uma imagem binária; 'area_minima' é o limiar de área em pixels.
# Pós-condição: Retorna uma lista de contornos ('rois_contornos') cuja área é maior ou igual a 'area_minima'.
def region_of_interest(img_bordas, area_minima):
    contornos, _ = cv2.findContours(img_bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total de contornos encontrados: {len(contornos)}")
    
    rois_contornos = [c for c in contornos if cv2.contourArea(c) >= area_minima]
    print(f"Número de regiões detectadas (área >= {area_minima}): {len(rois_contornos)}")
    return rois_contornos

# Função para calcular a assinatura normalizada de um contorno.
# Pré-condição: 'contorno' é um contorno válido retornado pelo OpenCV.
# Pós-condição: Retorna um vetor NumPy 1D (a assinatura da forma), com valores normalizados entre 0 e 1. Retorna None se o contorno for inválido.
def calcular_assinatura(contorno):
    M = cv2.moments(contorno)
    
    if M["m00"] == 0: return None
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    assinatura = [np.sqrt((p[0][0] - cx) ** 2 + (p[0][1] - cy) ** 2) for p in contorno]
    assinatura = np.array(assinatura, dtype=float)
    
    if np.max(assinatura) > 0: assinatura /= np.max(assinatura)
    return assinatura



# Função principal de classificação que determina a forma de uma ROI.
# Pré-condição: 'contorno' é um contorno válido; 'assinaturas_padrao' é um dicionário com as assinaturas ideais.
# Pós-condição: Retorna uma tupla (nome_da_forma, assinatura_calculada). 'nome_da_forma' é uma string ou None se não for reconhecida.
def classificar_placa(contorno, assinaturas_padrao, th_assinatura_circulo=0.015):
    assinatura_para_plotar = calcular_assinatura(contorno) # Sempre calculamos para poder visualizar
    
    # --- MÉTODO 1: Contagem de Vértices para Polígonos ---
    perimetro = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
    num_vertices = len(approx)
    print(f"--- Analisando ROI ---\nVértices encontrados (após aproximação): {num_vertices}")
 
    if num_vertices == 3 or (num_vertices >= 7 and num_vertices <= 9):
        print("--> Conclusão: Classificado como Octógono por contagem de vértices.")
        return "Placa de Regulamentacao", assinatura_para_plotar
 
    if num_vertices == 4:
        print("--> Conclusão: Classificado como Losango por contagem de vértices.")
        return "Placa de Advertencia", assinatura_para_plotar
     
    # --- MÉTODO 2 (Plano B): Assinatura da Forma para Curvas ---
    print(f"--> {num_vertices} vértices não corresponde a um polígono conhecido. Testando como círculo...")
    
    # Comparamos a assinatura apenas com o molde do círculo
    assinatura_circulo = assinaturas_padrao.get("Placa de Regulamentacao")
    if assinatura_circulo is not None:
        diferenca = np.mean((cv2.resize(assinatura_para_plotar, (100, 1)).flatten() - cv2.resize(assinatura_circulo, (100, 1)).flatten()) ** 2)
        print(f"Diferença para o círculo: {diferenca:.4f}")
        if diferenca < th_assinatura_circulo:
            print("--> Conclusão: Classificado como Círculo por baixa diferença de assinatura.")
            return "Placa de Regulamentacao", assinatura_para_plotar

    print("--> Não corresponde a nenhum padrão conhecido.")
    return None, assinatura_para_plotar


# Função para limpar a interface gráfica, retornando-a ao estado inicial.
# Pré-condição: Os widgets da GUI (lbl_original, etc.) devem existir e ser acessíveis.
# Pós-condição: Todos os painéis de imagem e textos de resultado são limpos.
def resetar_interface():
    labels_para_limpar = [
        lbl_original, lbl_cinza, lbl_clahe, lbl_bordas, 
        lbl_bordas_com_contornos, lbl_resultado_forma
    ]
    
    # Cria uma imagem de placeholder vazia para colocar nos labels
    placeholder = ImageTk.PhotoImage(Image.new("RGB", (350, 350), (240, 240, 240)))
    
    for label in labels_para_limpar:
        label.config(image=placeholder)
        label.image = placeholder
        
    # Limpa o gráfico da assinatura
    plotar_assinatura_em_label(None, lbl_assinatura, "Aguardando ROI")
    
    # Limpa o texto de resultado
    lbl_resultado_texto.config(text="Aguardando análise...")
    print("Interface resetada.")


# =============================================================================
# EXECUÇÃO PRINCIPAL DO SCRIPT
# =============================================================================
# Função principal que faz todo procedimento quando o usuário carrega uma imagem.
# Pré-condição: A interface gráfica e as assinaturas padrão devem estar inicializadas.
# Pós-condição: A interface é atualizada com os resultados do processamento e classificação da imagem.
def main():
    
    resetar_interface() # Reseta a interface a cada nova imagem para uma experiência limpa
    
    caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg")])
    if not caminho: 
        resetar_interface() # Reseta se o usuário cancelar a seleção
        return
    
    # Etapa 1: Pré-processamento e Segmentação
    img_original = cv2.imread(caminho)
    img_redim = redimensionar_imagem(img_original)
    img_cinza = converter_para_cinza(img_redim)
    img_clahe = aplicar_clahe(img_cinza)
    img_bordas = detectar_bordas(img_clahe)
    contornos_roi = region_of_interest(img_bordas, area_minima=500)

    # Etapa 2: Visualização dos Filtros
    exibir_imagem_no_label(img_redim, lbl_original)
    exibir_imagem_no_label(img_cinza, lbl_cinza)
    exibir_imagem_no_label(img_clahe, lbl_clahe)
    exibir_imagem_no_label(img_bordas, lbl_bordas)
    img_bordas_com_contornos = garantir_bgr(img_bordas.copy())
    cv2.drawContours(img_bordas_com_contornos, contornos_roi, -1, (0, 255, 0), 2)
    exibir_imagem_no_label(img_bordas_com_contornos, lbl_bordas_com_contornos)

    # Etapa 3: Análise de Forma
    img_final_com_deteccoes = img_redim.copy()
    resultados_finais = [] # Lista para guardar os resultados de cada ROI

    if not contornos_roi:
        messagebox.showinfo("Concluído", "Nenhuma região de interesse significativa foi encontrada.")
        lbl_resultado_texto.config(text="Nenhuma ROI encontrada.")
        return

    for i, contorno in enumerate(contornos_roi):
        forma_detectada, assinatura_atual = classificar_placa(contorno, assinaturas_padrao)
        
        # Mostra a análise interativa 
        img_roi_atual = img_redim.copy()
        cv2.drawContours(img_roi_atual, [contorno], -1, (255, 0, 0), 2)
        plotar_assinatura_em_label(assinatura_atual, lbl_assinatura, f"Assinatura ROI {i+1}")
        exibir_imagem_no_label(img_roi_atual, lbl_resultado_forma)
        
        resultado_loop = f"ROI {i+1}: {forma_detectada}" if forma_detectada else f"ROI {i+1}: Nao corresponde"
        resultados_finais.append(resultado_loop) # Adiciona resultado à lista
        
        if forma_detectada:
            cv2.drawContours(img_final_com_deteccoes, [contorno], -1, (0, 255, 0), 2)
        
        messagebox.showinfo("Análise de ROI", f"Analisando ROI {i+1} de {len(contornos_roi)}.\n\nResultado: {resultado_loop}\n\nClique em OK para continuar.")


    # Etapa 4: Apresentação Final dos Resultados
    exibir_imagem_no_label(img_final_com_deteccoes, lbl_resultado_forma)
    texto_final_formatado = f"Análise Concluída. {len(resultados_finais)} ROIs avaliadas:\n\n"
    texto_final_formatado += "\n".join(resultados_finais)
    lbl_resultado_texto.config(text=texto_final_formatado, justify=tk.LEFT)
    
    messagebox.showinfo("Concluído", "Análise de todas as ROIs finalizada!")


# =============================================================================
# INTERFACE GRÁFICA (GUI)
# =============================================================================

# Bloco principal que inicia a aplicação.
# Pré-condição: Nenhuma.
# Pós-condição: A interface gráfica é criada, as assinaturas padrão são calculadas e o programa entra em loop, aguardando a interação do usuário.
if __name__ == "__main__":

    # 1. Otimização: Calcula as assinaturas padrão UMA ÚNICA VEZ
    circulo_img = np.zeros((100, 100), dtype=np.uint8); cv2.circle(circulo_img, (50, 50), 45, 255, 1); c, _ = cv2.findContours(circulo_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); assinatura_circulo = calcular_assinatura(c[0])
    losango_img = np.zeros((100, 100), dtype=np.uint8); pontos_losango = np.array([[50, 5], [95, 50], [50, 95], [5, 50]], np.int32); cv2.polylines(losango_img, [pontos_losango], True, 255, 2); los_cont, _ = cv2.findContours(losango_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); assinatura_losango = calcular_assinatura(los_cont[0])
    pontos_oct = np.array([[31, 10], [69, 10], [90, 31], [90, 69], [69, 90], [31, 90], [10, 69], [10, 31]], np.int32); octogono_img = np.zeros((100, 100), dtype=np.uint8); cv2.polylines(octogono_img, [pontos_oct], True, 255, 2); oct_cont, _ = cv2.findContours(octogono_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); assinatura_octogono = calcular_assinatura(oct_cont[0])
    pontos_tri_inv = np.array([[10, 10], [90, 10], [50, 90]], np.int32); tri_inv_img = np.zeros((100, 100), dtype=np.uint8); cv2.polylines(tri_inv_img, [pontos_tri_inv], True, 255, 2); tri_inv_cont, _ = cv2.findContours(tri_inv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); assinatura_tri_inv = calcular_assinatura(tri_inv_cont[0])
    
    assinaturas_padrao = {
        "Placa de Regulamentacao": assinatura_circulo,
        "Placa de Advertencia": assinatura_losango,
    }

    # 2. Criação da Janela e Widgets
    root = tk.Tk()
    root.title("Trabalho 2 - PDI")
    root.geometry("1280x720") # Tamanho inicial


    # Estrutura de Rolagem (Canvas + Scrollbars)
    container = ttk.Frame(root)
    container.pack(fill=tk.BOTH, expand=True)
    main_canvas = tk.Canvas(container)
    #BARRA DE ROLAGEM HORIZONTAL & VERTICAL
    h_scrollbar = ttk.Scrollbar(container, orient="horizontal", command=main_canvas.xview)
    v_scrollbar = ttk.Scrollbar(container, orient="vertical", command=main_canvas.yview)
    main_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    # Posiciona os widgets na tela usando grid para melhor controle
    h_scrollbar.grid(row=1, column=0, sticky="ew")
    v_scrollbar.grid(row=0, column=1, sticky="ns")
    main_canvas.grid(row=0, column=0, sticky="nsew")
    
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    scrollable_frame = ttk.Frame(main_canvas)
    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Configuração do frame rolável para ajustar o tamanho do canvas
    def on_frame_configure(canvas):
        canvas.configure(scrollregion=canvas.bbox("all"))
        
    scrollable_frame.bind("<Configure>", lambda e: on_frame_configure(main_canvas))

    #Função para funcionar o scroll com a roda do mouse>>>>
    def _on_mousewheel(event):
        # Shift + Roda do Mouse = Scroll Horizontal
        if event.state & 0x0004:
            main_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        # Apenas Roda do Mouse = Scroll Vertical
        else:
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
    

    # 3. Adiciona o conteúdo da interface DENTRO do frame rolável
    content_frame = ttk.Frame(scrollable_frame, padding="10 10 10 10")
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    # Botões
    control_frame = ttk.Frame(content_frame)
    control_frame.pack(pady=(0, 10), fill=tk.X)
    control_frame.columnconfigure((0, 1), weight=1)
    btn_carregar = ttk.Button(control_frame, text="Carregar Imagem", command=main)
    btn_carregar.grid(row=0, column=0, padx=5, sticky="ew")
    btn_resetar = ttk.Button(control_frame, text="Resetar", command=lambda: None)
    btn_resetar.grid(row=0, column=1, padx=5, sticky="ew")

    # Painel de Pré-processamento
    frame_filtros = ttk.LabelFrame(content_frame, text="Pré-processamento", padding="10")
    frame_filtros.pack(fill=tk.X, anchor="nw") # Âncora no noroeste
    for i in range(5): frame_filtros.columnconfigure(i, weight=1)
    lbl_original = ttk.Label(frame_filtros); 
    lbl_original.grid(row=0, column=0, padx=5, pady=5); 
    ttk.Label(frame_filtros, text="Original").grid(row=1, column=0)
    
    lbl_cinza = ttk.Label(frame_filtros); 
    lbl_cinza.grid(row=0, column=1, padx=5, pady=5); 
    ttk.Label(frame_filtros, text="Escala de Cinza").grid(row=1, column=1)
    
    lbl_clahe = ttk.Label(frame_filtros); 
    lbl_clahe.grid(row=0, column=2, padx=5, pady=5); 
    ttk.Label(frame_filtros, text="CLAHE").grid(row=1, column=2)
    
    lbl_bordas = ttk.Label(frame_filtros); 
    lbl_bordas.grid(row=0, column=3, padx=5, pady=5); 
    ttk.Label(frame_filtros, text="Bordas (Canny)").grid(row=1, column=3)
    
    lbl_bordas_com_contornos = ttk.Label(frame_filtros); 
    lbl_bordas_com_contornos.grid(row=0, column=4, padx=5, pady=5); 
    ttk.Label(frame_filtros, text="Contornos Detectados").grid(row=1, column=4)

    # Painel de Análise
    frame_analise_container = ttk.Frame(content_frame)
    frame_analise_container.pack(pady=(10, 0), fill=tk.BOTH, expand=True, anchor="nw")
    frame_analise_container.columnconfigure(0, weight=1)
    frame_analise_container.columnconfigure(1, weight=1)
    
    frame_analise_visual = ttk.LabelFrame(frame_analise_container, text="Análise da ROI", padding="10")
    frame_analise_visual.grid(row=0, column=0, sticky="nsew")
    frame_analise_visual.columnconfigure((0, 1), weight=1)
    lbl_assinatura = ttk.Label(frame_analise_visual); lbl_assinatura.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
    lbl_resultado_forma = ttk.Label(frame_analise_visual); lbl_resultado_forma.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    
    frame_resultados_texto = ttk.LabelFrame(frame_analise_container, text="Resultados da Análise", padding="10")
    frame_resultados_texto.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
    lbl_resultado_texto = ttk.Label(frame_resultados_texto, text="Aguardando análise...", anchor="nw", justify=tk.LEFT, font=('Courier', 10))
    lbl_resultado_texto.pack(fill=tk.BOTH, expand=True)

    # 4. Configuração final
    btn_resetar.config(command=resetar_interface)
    resetar_interface()
    root.mainloop()