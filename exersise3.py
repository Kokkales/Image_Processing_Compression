import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import heapq
import os

# Ορισμός των πινάκων κβάντισης
Q10 = np.array([
    [80, 60, 50, 80, 120, 200, 255, 255],
    [55, 60, 70, 95, 130, 255, 255, 255],
    [70, 65, 80, 120, 200, 255, 255, 255],
    [70, 85, 110, 145, 255, 255, 255, 255],
    [90, 110, 185, 255, 255, 255, 255, 255],
    [120, 175, 255, 255, 255, 255, 255, 255],
    [245, 255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255, 255]
])

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image)

def save_image(image_array, output_path):
    image = Image.fromarray(np.uint8(image_array))
    image.save(output_path)

def block_splitting(image, block_size=8):
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(image[i:i+block_size, j:j+block_size])
    return blocks

def recombine_blocks(blocks, image_shape, block_size=8):
    h, w = image_shape
    image = np.zeros(image_shape)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            image[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return image

def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, q_table):
    return np.round(block / q_table).astype(np.int32)

def dequantize(block, q_table):
    return (block * q_table).astype(np.float32)

def zigzag_order(block):
    h, w = block.shape
    result = []
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(i+1):
                if j < h and i - j < w:
                    result.append(block[j, i - j])
        else:
            for j in range(i+1):
                if i - j < h and j < w:
                    result.append(block[i - j, j])
    return np.array(result)

def inverse_zigzag_order(arr, block_size=8):
    h, w = block_size, block_size
    block = np.zeros((h, w), dtype=np.float32)
    index = 0
    for i in range(h + w - 1):
        if i % 2 == 0:
            for j in range(i+1):
                if j < h and i - j < w:
                    block[j, i - j] = arr[index]
                    index += 1
        else:
            for j in range(i+1):
                if i - j < h and j < w:
                    block[i - j, j] = arr[index]
                    index += 1
    return block

class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(freqs):
    heap = [Node(sym, freq) for sym, freq in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]

def generate_codes(node, prefix="", codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = prefix
        generate_codes(node.left, prefix + "0", codebook)
        generate_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encoding(arr):
    freqs = Counter(arr)
    root = build_tree(freqs)
    codebook = generate_codes(root)
    encoded = ''.join([codebook[sym] for sym in arr])
    return encoded, codebook

def huffman_decoding(encoded, codebook):
    reverse_codebook = {v: k for k, v in codebook.items()}
    decoded = []
    current_code = ""
    for bit in encoded:
        current_code += bit
        if current_code in reverse_codebook:
            decoded.append(reverse_codebook[current_code])
            current_code = ""
    return np.array(decoded)

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))

def jpeg_compress(image_path, q_table):
    image = load_image(image_path)
    blocks = block_splitting(image)
    dct_blocks = [dct_2d(block) for block in blocks]
    quantized_blocks = [quantize(block, q_table) for block in dct_blocks]
    zigzag_blocks = [zigzag_order(block) for block in quantized_blocks]
    all_coefficients = np.concatenate(zigzag_blocks)
    encoded, codebook = huffman_encoding(all_coefficients)
    return encoded, codebook, image.shape, len(encoded) / len(all_coefficients), len(image.flatten()) * 8 / len(encoded)

def jpeg_decompress(encoded, codebook, image_shape, q_table):
    decoded = huffman_decoding(encoded, codebook)
    blocks = np.split(decoded, len(decoded) / 64)
    dequantized_blocks = [dequantize(inverse_zigzag_order(block), q_table) for block in blocks]
    idct_blocks = [idct_2d(block) for block in dequantized_blocks]
    reconstructed_image = recombine_blocks(idct_blocks, image_shape)
    return reconstructed_image

def main():
    images = ["./images/bridge.bmp", "./images/girlface.bmp", "./images/lighthouse.bmp"]
    q_tables = [Q10, Q50]
    q_table_names = ["Q10", "Q50"]

    results = {}

    for img in images:
        results[img] = {}
        for q_table, q_table_name in zip(q_tables, q_table_names):
            encoded, codebook, image_shape, avg_codeword_length, compression_ratio = jpeg_compress(img, q_table)
            reconstructed_image = jpeg_decompress(encoded, codebook, image_shape, q_table)
            original_image = load_image(img)
            psnr_value = psnr(original_image, reconstructed_image)
            results[img][q_table_name] = {
                "avg_codeword_length": avg_codeword_length,
                "compression_ratio": compression_ratio,
                "psnr": psnr_value
            }
            output_image_path = f"reconstructed_{os.path.splitext(img)[0]}_{q_table_name}.bmp"
            save_image(reconstructed_image, output_image_path)

    for img, res in results.items():
        print(f"Results for {img}:")
        for q_table_name, metrics in res.items():
            print(f"  {q_table_name}:")
            print(f"    Average codeword length: {metrics['avg_codeword_length']}")
            print(f"    Compression ratio: {metrics['compression_ratio']}")
            print(f"    PSNR: {metrics['psnr']}")

if __name__ == "__main__":
    main()