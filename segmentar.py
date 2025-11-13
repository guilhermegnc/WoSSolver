import cv2
import numpy as np

def segment_letters(image_path, debug=False):
    """
    Segmenta letras individuais de uma imagem estilo Scrabble
    Retorna lista de tiles em memória (sem salvar PNGs)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    original = img.copy()
    height, width = img.shape[:2]
    
    letter_regions = detect_by_projection(img, width, height, debug)

    if len(letter_regions) < 3:
        tile_regions = detect_tiles_precise(img, width, height, debug)
        if len(tile_regions) > 0:
            letter_regions = tile_regions

    if len(letter_regions) < 2:
        hsv_regions = detect_by_hsv(img, width, height, debug)
        if len(hsv_regions) > 0:
            letter_regions = hsv_regions

    if len(letter_regions) < 2:
        edge_regions = detect_by_edges(img, width, height, debug)
        if len(edge_regions) > 0:
            letter_regions = edge_regions

    if len(letter_regions) < 2:
        seg_regions = detect_by_color_segmentation(img, width, height, debug)
        if len(seg_regions) > 0:
            letter_regions = seg_regions
    
    # Ordenar da esquerda para direita
    letter_regions = sorted(letter_regions, key=lambda r: r['center_x'])
    
    # Extrair imagens dos TILES (usar warp quando disponível)
    segmented_letters = []
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for idx, region in enumerate(letter_regions):
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Se a região já contém uma imagem warpada (retificada), use-a
        if 'warped_image' in region and region['warped_image'] is not None:
            tile_color = region['warped_image']
            if len(tile_color.shape) == 3 and tile_color.shape[2] == 3:
                tile_gray = cv2.cvtColor(tile_color, cv2.COLOR_BGR2GRAY)
            else:
                tile_gray = tile_color.copy()
        else:
            # Fallback para recorte axis-aligned com padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, int(x + w + padding))
            y2 = min(height, int(y + h + padding))
            tile_color = original[y1:y2, x1:x2]
            tile_gray = gray_full[y1:y2, x1:x2]

        # Binarização para cada tile
        tile_binary = cv2.adaptiveThreshold(
            tile_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        segmented_letters.append({
            'image': tile_gray,
            'image_color': tile_color,
            'image_binary': tile_binary,
            'position': (x, y, w, h),
            'original_region': region,
            'index': idx
        })
    
    return segmented_letters, img


def detect_by_hsv(img, width, height, debug=False):
    """
    Método 1: Detecção por cor HSV (bordas roxas)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Múltiplos ranges para capturar variações de roxo
    masks = []
    
    # Roxo escuro
    masks.append(cv2.inRange(hsv, np.array([130, 40, 40]), np.array([160, 255, 255])))
    # Roxo médio
    masks.append(cv2.inRange(hsv, np.array([125, 30, 60]), np.array([155, 255, 200])))
    # Roxo claro/lavanda
    masks.append(cv2.inRange(hsv, np.array([135, 20, 80]), np.array([165, 150, 220])))
    
    # Combinar todas as máscaras
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    
    # Morfologia agressiva para conectar bordas
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return filter_contours(contours, width, height, min_area=300, debug_prefix="HSV")


def detect_by_edges(img, width, height, debug=False):
    """
    Método 2: Detecção por bordas Canny + Threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur para reduzir ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny com thresholds baixos
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilatar bordas
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)
    
    # Threshold Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    
    # Combinar
    combined = cv2.bitwise_or(edges, thresh_inv)
    
    # Fechar buracos
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return filter_contours(contours, width, height, min_area=400, debug_prefix="EDGES")


def detect_by_color_segmentation(img, width, height, debug=False):
    """
    Método 3: Segmentação por cor direta (LAB/BGR)
    Detecta os tiles completos (incluindo bordas roxas)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Estratégia: Detectar o fundo roxo claro e inverter
    # Isso pegará todo o tile (borda roxa + interior claro + letra)
    
    # Fundo roxo claro
    lower_bg = np.array([120, 10, 150])
    upper_bg = np.array([170, 100, 255])
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Inverter para pegar os tiles completos (não-fundo)
    tiles_mask = cv2.bitwise_not(bg_mask)
    
    # Operações morfológicas para fechar os tiles completamente
    # Usar kernel maior para garantir que borda + interior sejam unidos
    kernel_large = np.ones((9, 9), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # Primeiro limpar pequenos ruídos
    tiles_mask = cv2.morphologyEx(tiles_mask, cv2.MORPH_OPEN, kernel_medium, iterations=1)
    
    # Depois fechar grandes gaps (unir bordas com interior)
    tiles_mask = cv2.morphologyEx(tiles_mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    
    # Dilatar levemente para garantir que pegamos toda a borda
    tiles_mask = cv2.dilate(tiles_mask, kernel_medium, iterations=2)
    
    contours, _ = cv2.findContours(tiles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Usar filtro simples que funciona + remoção de duplicatas
    return filter_and_deduplicate(contours, width, height, min_area=500, debug_prefix="SEG")


def detect_by_projection(img, width, height, debug=False):
    """
    Método 4: Projeção horizontal - assume tiles alinhados em linha
    Versão revisada: detecta separações verticais para isolar tiles INDIVIDUAIS
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold adaptativo para melhor separação
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Projeção vertical (soma por coluna) - encontra separadores entre tiles
    vertical_proj = np.sum(binary, axis=0)
    
    # Normalizar
    if np.max(vertical_proj) > 0:
        vertical_proj_norm = (vertical_proj / np.max(vertical_proj)) * 255
    else:
        vertical_proj_norm = vertical_proj
    
    # Encontrar vales (colunas vazias = separadores) com threshold mais AGRESSIVO
    # Isso força encontrar gaps menores entre tiles
    threshold = np.mean(vertical_proj) * 0.08  # Reduzido de 0.15 (mais sensível)
    in_tile = False
    tile_start = 0
    regions = []
    
    for i, val in enumerate(vertical_proj):
        if val > threshold and not in_tile:
            tile_start = i
            in_tile = True
        elif val <= threshold and in_tile:
            # Final de uma tile
            tile_end = i
            w = tile_end - tile_start
            
            # Estimar altura (assumir que tiles são aproximadamente quadrados)
            h = int(w * 1.2)
            y = max(0, (height - h) // 2)
            
            if w > 10:  # Largura mínima bem pequena agora
                regions.append({
                    'x': tile_start,
                    'y': y,
                    'w': w,
                    'h': min(h, height - y),
                    'contour': None,
                    'area': w * h,
                    'center_x': tile_start + w/2
                })
            
            in_tile = False
    
    # Se ainda estiver dentro de uma tile no final, adicionar a última
    if in_tile:
        tile_end = width
        w = tile_end - tile_start
        h = int(w * 1.2)
        y = max(0, (height - h) // 2)
        if w > 10:
            regions.append({
                'x': tile_start,
                'y': y,
                'w': w,
                'h': min(h, height - y),
                'contour': None,
                'area': w * h,
                'center_x': tile_start + w/2
            })
    
    return regions


def extract_warped_tile(img, box_pts, padding=6):
    """
    Ordena os 4 pontos, calcula dimensão destino e retorna a imagem warpada
    e o bbox axis-aligned (x,y,w,h) referente à posição aproximada na imagem
    """
    box = np.array(box_pts, dtype="float32")

    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    rect = order_points(box)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB)) + padding * 2

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB)) + padding * 2

    dst = np.array([
        [padding, padding],
        [maxWidth - padding - 1, padding],
        [maxWidth - padding - 1, maxHeight - padding - 1],
        [padding, maxHeight - padding - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    x = int(max(0, np.min(box[:, 0]) - padding))
    y = int(max(0, np.min(box[:, 1]) - padding))

    return warp, (x, y, maxWidth, maxHeight), rect


def detect_tiles_precise(img, width, height, debug=False):
    """
    Detecta cada tile individual separando componentes conectadas.
    Usa thresholding + erosão para separar tiles que estão perto.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold simples + adaptativo para melhor separação
    _, binary_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 3)
    
    # Combinar os dois
    binary = cv2.bitwise_and(binary_simple, binary_adapt)
    
    # Erosão para separar componentes conectadas (cria gaps entre letras)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary, kernel, iterations=1)
    
    # Dilatação para recuperar o tamanho
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # Filtro de área mínima reduzido
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtros de tamanho
        if w < 15 or h < 15:
            continue
        
        aspect = w / float(h) if h > 0 else 0
        # Tiles devem ser aproximadamente quadrados ou ligeiramente retangulares
        if aspect < 0.4 or aspect > 3.0:
            continue
        
        # Não deve ocupar mais de 40% da largura total
        if w > width * 0.4:
            continue
        
        regions.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'contour': cnt,
            'area': area,
            'center_x': x + w / 2
        })
    
    # Ordenar por posição horizontal (esquerda para direita)
    regions = sorted(regions, key=lambda r: r['x'])
    
    return regions


def filter_contours(contours, width, height, min_area=300, debug_prefix=""):
    """
    Filtra contornos com critérios muito flexíveis
    """
    regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Tamanho mínimo
        if w < 15 or h < 15:
            continue
        
        # Aspect ratio muito permissivo
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
        # Não deve ocupar toda a imagem
        if w > width * 0.95 or h > height * 0.95:
            continue
        
        regions.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'contour': contour,
            'area': area,
            'center_x': x + w/2
        })
    
    return regions


def filter_and_deduplicate(contours, width, height, min_area=300, debug_prefix=""):
    """
    Filtra contornos com critérios flexíveis e remove duplicatas
    Baseado no método que funciona + remoção de sobreposição
    """
    # PASSO 1: Filtrar com critérios básicos (que já funcionam)
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Tamanho mínimo
        if w < 15 or h < 15:
            continue
        
        # Aspect ratio muito permissivo
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
        # Não deve ocupar toda a imagem
        if w > width * 0.95 or h > height * 0.95:
            continue
        
        candidates.append({
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'contour': contour,
            'area': area,
            'center_x': x + w/2,
            'center_y': y + h/2
        })
    
    if not candidates:
        return []
    
    # PASSO 2: Remover duplicatas (regiões muito sobrepostas)
    # Ordenar por área (maiores primeiro para manter os melhores)
    candidates.sort(key=lambda c: c['area'], reverse=True)
    
    unique_regions = []
    duplicates = 0
    
    for candidate in candidates:
        is_duplicate = False
        
        for existing in unique_regions:
            # Calcular sobreposição
            x_left = max(candidate['x'], existing['x'])
            x_right = min(candidate['x'] + candidate['w'], existing['x'] + existing['w'])
            y_top = max(candidate['y'], existing['y'])
            y_bottom = min(candidate['y'] + candidate['h'], existing['y'] + existing['h'])
            
            # Calcular área de interseção
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                
                # Calcular porcentagem em relação à menor região
                smaller_area = min(candidate['area'], existing['area'])
                overlap_percent = (intersection / smaller_area) * 100
                
                # Se sobreposição > 70%, considerar duplicata
                if overlap_percent > 70:
                    is_duplicate = True
                    duplicates += 1
                    break
        
        if not is_duplicate:
            unique_regions.append(candidate)
    
    return unique_regions


def get_tile_color_images(segmented_letters):
    """
    Retorna dicionário com as imagens color dos tiles em memória
    Formato: {'letra_1_tile_color': imagem_bgr, 'letra_2_tile_color': imagem_bgr, ...}
    """
    tile_dict = {}
    
    for letter_data in segmented_letters:
        idx = letter_data['index']
        if 'image_color' in letter_data and letter_data['image_color'] is not None:
            key = f"letra_{idx + 1}_tile_color"
            tile_dict[key] = letter_data['image_color']
    
    return tile_dict


if __name__ == "__main__":
    image_path = "1.png"
    
    try:
        segmented_letters, original_img = segment_letters(image_path, debug=False)
        
        if len(segmented_letters) == 0:
            print("❌ Nenhum tile detectado!")
        else:
            tile_images = get_tile_color_images(segmented_letters)
            print(f"✅ Sucesso! {len(segmented_letters)} tiles encontrados")
            print(f"   Imagens disponíveis: {list(tile_images.keys())}")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")