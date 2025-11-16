"""
This script provides functions to segment individual letters from an image,
particularly tailored for Scrabble-like letter tiles.
"""
import cv2
import numpy as np

def segment_letters(image_path, debug=False):
    """
    Segments individual letters from a Scrabble-style image.
    Returns a list of tiles in memory (without saving PNGs).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    original_image = img.copy()
    height, width = img.shape[:2]
    
    # Attempt different detection methods in order of preference
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
    
    # Sort from left to right
    letter_regions = sorted(letter_regions, key=lambda r: r['center_x'])
    
    # Extract TILE images (use warp when available)
    segmented_letters = []
    full_gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for idx, region in enumerate(letter_regions):
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # If the region already contains a warped (rectified) image, use it
        if 'warped_image' in region and region['warped_image'] is not None:
            tile_color = region['warped_image']
            if len(tile_color.shape) == 3 and tile_color.shape[2] == 3:
                tile_gray = cv2.cvtColor(tile_color, cv2.COLOR_BGR2GRAY)
            else:
                tile_gray = tile_color.copy()
        else:
            # Fallback to axis-aligned crop with padding
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, int(x + w + padding))
            y2 = min(height, int(y + h + padding))
            tile_color = original_image[y1:y2, x1:x2]
            tile_gray = full_gray_image[y1:y2, x1:x2]

        # Binarization for each tile
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
    Method 1: HSV color detection (purple borders).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Multiple ranges to capture variations of purple
    masks = []
    
    # Dark purple
    masks.append(cv2.inRange(hsv, np.array([130, 40, 40]), np.array([160, 255, 255])))
    # Medium purple
    masks.append(cv2.inRange(hsv, np.array([125, 30, 60]), np.array([155, 255, 200])))
    # Light purple/lavender
    masks.append(cv2.inRange(hsv, np.array([135, 20, 80]), np.array([165, 150, 220])))
    
    # Combine all masks
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    
    # Aggressive morphology to connect edges
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return filter_contours(contours, width, height, min_area=300, debug_prefix="HSV")


def detect_by_edges(img, width, height, debug=False):
    """
    Method 2: Canny edge detection + Threshold.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny with low thresholds
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)
    
    # Otsu's Threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_inv = cv2.bitwise_not(thresh)
    
    # Combine
    combined = cv2.bitwise_or(edges, thresh_inv)
    
    # Close holes
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return filter_contours(contours, width, height, min_area=400, debug_prefix="EDGES")


def detect_by_color_segmentation(img, width, height, debug=False):
    """
    Method 3: Direct color segmentation (LAB/BGR).
    Detects the complete tiles (including purple borders).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Strategy: Detect the light purple background and invert.
    # This will get the whole tile (purple border + light interior + letter).
    
    # Light purple background
    lower_bg = np.array([120, 10, 150])
    upper_bg = np.array([170, 100, 255])
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    # Invert to get the complete tiles (non-background)
    tiles_mask = cv2.bitwise_not(bg_mask)
    
    # Morphological operations to completely close the tiles
    # Use a larger kernel to ensure the border and interior are joined
    kernel_large = np.ones((9, 9), np.uint8)
    kernel_medium = np.ones((5, 5), np.uint8)
    
    # First, clean up small noises
    tiles_mask = cv2.morphologyEx(tiles_mask, cv2.MORPH_OPEN, kernel_medium, iterations=1)
    
    # Then close large gaps (join borders with interior)
    tiles_mask = cv2.morphologyEx(tiles_mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    
    # Dilate slightly to ensure we get the entire border
    tiles_mask = cv2.dilate(tiles_mask, kernel_medium, iterations=2)
    
    contours, _ = cv2.findContours(tiles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use a simple filter that works + duplicate removal
    return filter_and_deduplicate(contours, width, height, min_area=500, debug_prefix="SEG")


def detect_by_projection(img, width, height, debug=False):
    """
    Method 4: Horizontal projection - assumes tiles are aligned in a row.
    Revised version: detects vertical separations to isolate INDIVIDUAL tiles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold for better separation
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Vertical projection (sum by column) - finds separators between tiles
    vertical_proj = np.sum(binary, axis=0)
    
    # Normalize
    if np.max(vertical_proj) > 0:
        vertical_proj_norm = (vertical_proj / np.max(vertical_proj)) * 255
    else:
        vertical_proj_norm = vertical_proj
    
    # Find valleys (empty columns = separators) with a more AGGRESSIVE threshold.
    # This forces finding smaller gaps between tiles.
    threshold = np.mean(vertical_proj) * 0.08  # Reduced from 0.15 (more sensitive)
    in_tile = False
    tile_start = 0
    regions = []
    
    for i, val in enumerate(vertical_proj):
        if val > threshold and not in_tile:
            tile_start = i
            in_tile = True
        elif val <= threshold and in_tile:
            # End of a tile
            tile_end = i
            w = tile_end - tile_start
            
            # Estimate height (assume tiles are approximately square)
            h = int(w * 1.2)
            y = max(0, (height - h) // 2)
            
            if w > 10:  # Very small minimum width now
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
    
    # If still inside a tile at the end, add the last one
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
    Sorts the 4 points, calculates the destination dimension, returns the warped image
    and the axis-aligned bbox (x,y,w,h) for the approximate position in the image.
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
    Detects each individual tile by separating connected components.
    Uses thresholding + erosion to separate tiles that are close.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple + adaptive threshold for better separation
    _, binary_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 3)
    
    # Combine the two
    binary = cv2.bitwise_and(binary_simple, binary_adapt)
    
    # Erosion to separate connected components (creates gaps between letters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary, kernel, iterations=1)
    
    # Dilation to recover the size
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # Reduced minimum area filter
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Size filters
        if w < 15 or h < 15:
            continue
        
        aspect = w / float(h) if h > 0 else 0
        # Tiles should be approximately square or slightly rectangular
        if aspect < 0.4 or aspect > 3.0:
            continue
        
        # Should not occupy more than 40% of the total width
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
    
    # Sort by horizontal position (left to right)
    regions = sorted(regions, key=lambda r: r['x'])
    
    return regions


def filter_contours(contours, width, height, min_area=300, debug_prefix=""):
    """
    Filters contours with very flexible criteria.
    """
    regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum size
        if w < 15 or h < 15:
            continue
        
        # Very permissive aspect ratio
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
        # Should not occupy the entire image
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
    Filters contours with flexible criteria and removes duplicates
    based on overlap.
    """
    # STEP 1: Filter with basic criteria
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum size
        if w < 15 or h < 15:
            continue
        
        # Very permissive aspect ratio
        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
        # Should not occupy the entire image
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
    
    # STEP 2: Remove duplicates (highly overlapping regions)
    # Sort by area (largest first to keep the best ones)
    candidates.sort(key=lambda c: c['area'], reverse=True)
    
    unique_regions = []
    
    for candidate in candidates:
        is_duplicate = False
        
        for existing in unique_regions:
            # Calculate overlap
            x_left = max(candidate['x'], existing['x'])
            x_right = min(candidate['x'] + candidate['w'], existing['x'] + existing['w'])
            y_top = max(candidate['y'], existing['y'])
            y_bottom = min(candidate['y'] + candidate['h'], existing['y'] + existing['h'])
            
            # Calculate intersection area
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                
                # Calculate percentage relative to the smaller region
                smaller_area = min(candidate['area'], existing['area'])
                if smaller_area > 0:
                    overlap_percent = (intersection / smaller_area) * 100
                    
                    # If overlap > 70%, consider it a duplicate
                    if overlap_percent > 70:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            unique_regions.append(candidate)
    
    return unique_regions


def get_tile_color_images(segmented_letters):
    """
    Returns a dictionary with the color images of the tiles in memory.
    Format: {'letter_1_tile_color': bgr_image, 'letter_2_tile_color': bgr_image, ...}
    """
    tile_dict = {}
    
    for letter_data in segmented_letters:
        idx = letter_data['index']
        if 'image_color' in letter_data and letter_data['image_color'] is not None:
            key = f"letter_{idx + 1}_tile_color"
            tile_dict[key] = letter_data['image_color']
    
    return tile_dict


if __name__ == "__main__":
    image_path = "1.png"
    
    try:
        segmented_letters, original_img = segment_letters(image_path, debug=False)
        
        if not segmented_letters:
            print("❌ No tiles detected!")
        else:
            tile_images = get_tile_color_images(segmented_letters)
            print(f"✅ Success! {len(segmented_letters)} tiles found.")
            print(f"   Available images: {list(tile_images.keys())}")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
