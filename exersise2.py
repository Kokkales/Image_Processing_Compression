import cv2
import numpy as np
import heapq
import pickle
from collections import Counter
import os
import shutil

# Define the path to the folder
folder_path = './resultsTwo'

# Check if the folder exists
if os.path.exists(folder_path):
    # Iterate over the contents of the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if it is a file
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
else:
    print(f'The folder {folder_path} does not exist.')

# Your code starts here
print("Folder contents deleted. Starting the main code.")
# Add your main code logic here

class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def buildHuffmanTree(frequencies):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in enumerate(frequencies) if freq > 0] #arxikopoiise tous komvous
    heapq.heapify(heap)

    while len(heap) > 1: #while more than one nodes exist in the tree, till reach the root
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.frequency + right.frequency) #new node -> αθροισμα αριστερού και δεξιού
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0] #root

def calculateFrequencies(img):
    frequencies = np.zeros(256, dtype=int)
    for value in img.flatten():
        frequencies[value] += 1
    return frequencies


# take the tree, χρησημοποίησα chat gpt εδώ για να το πετύχω
def createHuffmanCodes(node, code='', codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = code
        createHuffmanCodes(node.left, code + '0', codebook)
        createHuffmanCodes(node.right, code + '1', codebook)
    return codebook

def encodeImage(img, codes):
    encodedImg = ''
    for value in img.flatten():
        encodedImg += codes[value]
    return encodedImg

def saveEncodedImage(encodedImg, filename):
    with open(filename, 'wb') as f:
        pickle.dump(encodedImg, f)

def savHuffmanCodes(codes, filename):
    with open(filename, 'wb') as f:
        pickle.dump(codes, f)

def calculateAverageCodeLength(codes, frequencies):
    totalSymbols = sum(frequencies)
    weightedSum = sum(len(code) * frequencies[symbol] for symbol, code in codes.items())
    return weightedSum / totalSymbols

def calculateEntropy(frequencies):
    totalSymbols = sum(frequencies)
    entropy = -sum((freq / totalSymbols) * np.log2(freq / totalSymbols) for freq in frequencies if freq > 0)
    return entropy

def calculateCompressionRatio(originalSize, encodedSize):
    return originalSize / encodedSize

def printResults(name, codes, avgLength, entropy, compressionRatio):
    name = name.replace('.bmp', '')
    print(name)  # Output: bridge
    for symbol, code in codes.items():
        with open(f'./resultsTwo/{name}_brightness.txt', 'a', encoding='utf-8') as file:
            # Write the formatted strings to the file
            file.write(f"  Brightness {symbol}: {code}\n")
        # print(f"  Brightness {symbol}: {code}")
    with open(f'./resultsTwo/output.txt', 'a', encoding='utf-8') as file:
        # Write the formatted strings to the file
        file.write(f"{name}----------------------------\n")
        file.write(f"1) Average encoded word length: {avgLength}\n")
        file.write(f"2) Entropy: {entropy}\n")
        file.write(f"3) Λόγος συμπίεσης: {compressionRatio}\n")
    print()


# images
bridgeImg = cv2.imread('./images/bridge.bmp', cv2.IMREAD_GRAYSCALE)
girlfaceImg = cv2.imread('./images/girlface.bmp', cv2.IMREAD_GRAYSCALE)
lighthouseImg = cv2.imread('./images/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# frequencies for each φωτεινότητα
bridgeFreq = calculateFrequencies(bridgeImg)
girlfaceFreq = calculateFrequencies(girlfaceImg)
lighthouseFreq = calculateFrequencies(lighthouseImg)

# Huffman tree
bridgeTree = buildHuffmanTree(bridgeFreq)
girlfaceTree = buildHuffmanTree(girlfaceFreq)
lighthouseTree = buildHuffmanTree(lighthouseFreq)

# Hufffman codes
bridgeCodes = createHuffmanCodes(bridgeTree)
girlfaceCodes = createHuffmanCodes(girlfaceTree)
lighthouseCodes = createHuffmanCodes(lighthouseTree)

# Image Encoding
bridgeEncoded = encodeImage(bridgeImg, bridgeCodes)
girlfaceEncoded = encodeImage(girlfaceImg, girlfaceCodes)
lighthouseEncoded = encodeImage(lighthouseImg, lighthouseCodes)

# save encoded image
path='./resultsTwo/'
saveEncodedImage(bridgeEncoded, path+'bridgeEncoded.bin')
saveEncodedImage(girlfaceEncoded, path+ 'girlfaceEncoded.bin')
saveEncodedImage(lighthouseEncoded, path+ 'lighthouseEncoded.bin')

# Αποθήκευση του πίνακα κωδίκων Huffman
savHuffmanCodes(bridgeCodes, path+ 'bridgeCodes.pkl')
savHuffmanCodes(girlfaceCodes, path+ 'girlfaceCodes.pkl')
savHuffmanCodes(lighthouseCodes, path+ 'lighthouseCodes.pkl')

# Υπολογισμός του μέσου μήκους κωδικής λέξης
bridgeAvgLength = calculateAverageCodeLength(bridgeCodes, bridgeFreq)
girlfaceAvgLength = calculateAverageCodeLength(girlfaceCodes, girlfaceFreq)
lighthouseAvgLength = calculateAverageCodeLength(lighthouseCodes, lighthouseFreq)

# Υπολογισμός της εντροπίας
bridgeEntropy = calculateEntropy(bridgeFreq)
girlfaceEntropy = calculateEntropy(girlfaceFreq)
lighthouseEntropy = calculateEntropy(lighthouseFreq)

# Υπολογισμός του λόγου συμπίεσης
bridgeOriginalSize = bridgeImg.size * 8
girlfaceOriginalSize = girlfaceImg.size * 8
lighthouseOriginalSize = lighthouseImg.size * 8

bridgeEncodedSize = len(bridgeEncoded)
girlfaceEncodedSize = len(girlfaceEncoded)
lighthouseEncodedSize = len(lighthouseEncoded)

bridgeCompressionRatio = calculateCompressionRatio(bridgeOriginalSize, bridgeEncodedSize)
girlfaceCompressionRatio = calculateCompressionRatio(girlfaceOriginalSize, girlfaceEncodedSize)
lighthouseCompressionRatio = calculateCompressionRatio(lighthouseOriginalSize, lighthouseEncodedSize)


printResults("bridge.bmp", bridgeCodes, bridgeAvgLength, bridgeEntropy, bridgeCompressionRatio)
printResults("girlface.bmp", girlfaceCodes, girlfaceAvgLength, girlfaceEntropy, girlfaceCompressionRatio)
printResults("lighthouse.bmp", lighthouseCodes, lighthouseAvgLength, lighthouseEntropy, lighthouseCompressionRatio)
