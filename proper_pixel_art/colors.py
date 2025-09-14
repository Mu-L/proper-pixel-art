from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np
from proper_pixel_art import utils

def get_cell_color(cell_pixels: np.ndarray) -> tuple[int,int,int]:
    """
    cell_pixels: shape (height_cell, width_cell, 3), dtype=uint8
    returns the most frequent RGB tuple in the cell_pixels block.
    """
    # flatten to tuple of pixel values
    flat = list(map(tuple, cell_pixels.reshape(-1, 3)))
    cell_color = Counter(flat).most_common(1)[0][0]
    return cell_color

def palette_img(img: Image.Image, num_colors: int = 16, quantize_method: int = 1) -> Image.Image:
    rbg_img = utils.clamp_alpha(img, mode='RGB')
    paletted = rbg_img.quantize(colors=num_colors, method=quantize_method)
    return paletted

def make_background_transparent(image: Image.Image) -> Image.Image:
    """Make image background transparent."""
    im = image.convert("RGBA")
    corners = [(0, 0), (im.width-1, 0), (0, im.height-1), (im.width-1, im.height-1)]
    for corner_x, corner_y in corners:
        fill_color = (0, 0, 0, 0)
        ImageDraw.floodfill(im, (corner_x, corner_y), fill_color, thresh=0)
    return im

def main():
    img_path = Path.cwd() / "assets" / "blob" / "blob.png"
    img = Image.open(img_path).convert("RGBA")
    paletted = palette_img(img)
    paletted.show()

if __name__ == "__main__":
    main()
