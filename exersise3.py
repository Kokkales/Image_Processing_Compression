import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
import huffman
import os
from collections import Counter

# A.
#  masks
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

def loadImage(imagePath):
    image = Image.open(imagePath).convert('L')
    return np.array(image)

def saveImage(imageArray, outputPath):
    image = Image.fromarray(np.uint8(imageArray))
    image.save(outputPath)

def blockSplitting(image, blockSize=8):
    h, w = image.shape
    blocks = []
    for i in range(0, h, blockSize):
        for j in range(0, w, blockSize):
            blocks.append(image[i:i+blockSize, j:j+blockSize])
    return blocks

def recombineBlocks(blocks, imageShape, blockSize=8):
    h, w = imageShape
    image = np.zeros(imageShape)
    idx = 0
    for i in range(0, h, blockSize):
        for j in range(0, w, blockSize):
            image[i:i+blockSize, j:j+blockSize] = blocks[idx]
            idx += 1
    return image

def dct2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, qTable):
    return np.round(block / qTable).astype(np.int32)

def dequantize(block, qTable):
    return (block * qTable).astype(np.float32)

def zigzagOrder(block):
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

def inverseZigzagOrder(arr, blockSize=8):
    h, w = blockSize, blockSize
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

# B.
def jpegCompress(imagePath, qTable):
    image = loadImage(imagePath)
    blocks = blockSplitting(image)
    dctBlocks = [dct2d(block) for block in blocks]
    quantizedBlocks = [quantize(block, qTable) for block in dctBlocks]
    zigzagBlocks = [zigzagOrder(block) for block in quantizedBlocks]
    allCoefficients = np.concatenate(zigzagBlocks)

    # Use the huffman library for encoding
    huff = huffman.codebook(Counter(allCoefficients).items())
    encoded = ''.join(huff[sym] for sym in allCoefficients)

    return encoded, huff, image.shape, len(encoded) / len(ιμαγ), len(image.flatten()) * 8 / len(encoded)

def jpegDecompress(encoded, huff, imageShape, qTable):
    # Decode using the huffman library
    reverseHuff = {v: k for k, v in huff.items()}
    decoded = []
    currentCode = ""
    for bit in encoded:
        currentCode += bit
        if currentCode in reverseHuff:
            decoded.append(reverseHuff[currentCode])
            currentCode = ""
    decoded = np.array(decoded)

    blocks = np.split(decoded, len(decoded) / 64)
    dequantizedBlocks = [dequantize(inverseZigzagOrder(block), qTable) for block in blocks]
    idctBlocks = [idct2d(block) for block in dequantizedBlocks]
    reconstructedImage = recombineBlocks(idctBlocks, imageShape)
    return reconstructedImage

# Γ. PNSR
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    pixelMax = 255.0
    return 20 * np.log10(pixelMax / np.sqrt(mse))

images = ["./images/bridge.bmp", "./images/girlface.bmp", "./images/lighthouse.bmp"]
qTables = [Q10, Q50]
qTableNames = ["Q10", "Q50"]

results = {}

for img in images:
    results[img] = {}
    originalImage = loadImage(img)


    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Original Image - {os.path.basename(img)}")
    plt.imshow(originalImage, cmap='gray')
    plt.axis('off')

    for i, (qTable, qTableName) in enumerate(zip(qTables, qTableNames), start=2):
        encoded, huff, imageShape, avgCodewordLength, compressionRatio = jpegCompress(img, qTable)
        reconstructedImage = jpegDecompress(encoded, huff, imageShape, qTable)
        psnrValue = psnr(originalImage, reconstructedImage)
        results[img][qTableName] = {
            "avgCodewordLength": avgCodewordLength,
            "compressionRatio": compressionRatio,
            "psnr": psnrValue
        }
        outputImagePath = f"./resultsThree/reconstructed_{os.path.splitext(os.path.basename(img))[0]}_{qTableName}.bmp"
        saveImage(reconstructedImage, outputImagePath)

        #decoded εικόνες
        plt.subplot(1, 3, i)
        plt.title(f"Decoded Image - {qTableName}")
        plt.imshow(reconstructedImage, cmap='gray')
        plt.axis('off')
    plt.savefig(f'./resultsThree/all{os.path.splitext(os.path.basename(img))[0]}.jpg')
    plt.show()

for img, res in results.items():
    print(f"Results for {img}:")
    for qTableName, metrics in res.items():
        print(f"  {qTableName}:")
        print(f"    Average codeword length: {metrics['avgCodewordLength']}")
        print(f"    Compression ratio: {metrics['compressionRatio']}")
        print(f"    PSNR: {metrics['psnr']}")
