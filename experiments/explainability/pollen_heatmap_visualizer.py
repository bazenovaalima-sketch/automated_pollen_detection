import cv2
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Change these paths to match your files in Google Colab
INPUT_IMAGE_PATH = "1.png"  
OUTPUT_IMAGE_PATH = "pollen_heatmap_ready.png"

# Style parameters
IMAGE_SIZE = 640         
HEADER_HEIGHT = 65       
COLUMN_GAP = 2           
OUTER_BORDER = 10        

TITLES = ["Original Image", "YOLOv8l (EigenCAM)", "YOLOv26l (EigenCAM)"]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def resize_and_pad(img, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    """Resizes image keeping aspect ratio, padding the rest with white."""
    h, w = img.shape[:2]
    sh, sw = target_size
    aspect = w / h
    
    if aspect > sw / sh:
        new_w = sw
        new_h = int(new_w / aspect)
    else:
        new_h = sh
        new_w = int(new_h * aspect)
    
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final_img = np.ones((sh, sw, 3), dtype=np.uint8) * 255
    
    x_offset = (sw - new_w) // 2
    y_offset = (sh - new_h) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled_img
    
    return final_img

def process_single_heatmap(image_path):
    """Splits the strip, crops stretched heatmaps, and formats the final grid."""
    strip = cv2.imread(image_path)
    if strip is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    h, w = strip.shape[:2]
    part_w = w // 3  
    
    # Extract sections
    original_img = strip[:, 0:part_w]
    yolo8_heatmap = strip[:, part_w:2*part_w]
    yolo26_heatmap = strip[:, 2*part_w:3*part_w]
    
    # Restore aspect ratio (fixing vertical stretching)
    yolo8_fixed = yolo8_heatmap[:, 0:part_w]
    yolo26_fixed = yolo26_heatmap[:, 0:part_w]
    
    # Uniform padding
    p1 = resize_and_pad(original_img)
    p2 = resize_and_pad(yolo8_fixed)
    p3 = resize_and_pad(yolo26_fixed)
    
    divider = np.ones((IMAGE_SIZE, COLUMN_GAP, 3), dtype=np.uint8) * 255
    return np.hstack((p1, divider, p2, divider, p3))

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    try:
        grid = process_single_heatmap(INPUT_IMAGE_PATH)
        total_w = grid.shape[1]
        
        # Header
        header = np.ones((HEADER_HEIGHT, total_w, 3), dtype=np.uint8) * 255
        centers = [IMAGE_SIZE//2, IMAGE_SIZE+COLUMN_GAP+IMAGE_SIZE//2, 2*(IMAGE_SIZE+COLUMN_GAP)+IMAGE_SIZE//2]
        
        for title, cx in zip(TITLES, centers):
            size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(header, title, (cx - size[0]//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            
        final = np.vstack((header, grid))
        final = cv2.copyMakeBorder(final, OUTER_BORDER, OUTER_BORDER, OUTER_BORDER, OUTER_BORDER, 
                                  cv2.BORDER_CONSTANT, value=(255,255,255))
        
        cv2.imwrite(OUTPUT_IMAGE_PATH, final)
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
