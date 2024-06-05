import cv2
import numpy as np
import heapq
import pickle
from collections import Counter

class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency

def calculate_frequencies(img):
    frequencies = np.zeros(256, dtype=int)
    for value in img.flatten():
        frequencies[value] += 1
    return frequencies

def build_huffman_tree(frequencies):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in enumerate(frequencies) if freq > 0]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def create_huffman_codes(node, code='', codebook={}):
    if node is not None:
        if node.symbol is not None:
            codebook[node.symbol] = code
        create_huffman_codes(node.left, code + '0', codebook)
        create_huffman_codes(node.right, code + '1', codebook)
    return codebook

def encode_image(img, codes):
    encoded_img = ''
    for value in img.flatten():
        encoded_img += codes[value]
    return encoded_img

def save_encoded_image(encoded_img, filename):
    with open(filename, 'wb') as f:
        pickle.dump(encoded_img, f)

def save_huffman_codes(codes, filename):
    with open(filename, 'wb') as f:
        pickle.dump(codes, f)

def calculate_average_code_length(codes, frequencies):
    total_symbols = sum(frequencies)
    weighted_sum = sum(len(code) * frequencies[symbol] for symbol, code in codes.items())
    return weighted_sum / total_symbols

def calculate_entropy(frequencies):
    total_symbols = sum(frequencies)
    entropy = -sum((freq / total_symbols) * np.log2(freq / total_symbols) for freq in frequencies if freq > 0)
    return entropy

def calculate_compression_ratio(original_size, encoded_size):
    return original_size / encoded_size

# Φόρτωση των εικόνων σε ασπρόμαυρο (grayscale)
bridge_img = cv2.imread('./images/bridge.bmp', cv2.IMREAD_GRAYSCALE)
girlface_img = cv2.imread('./images/girlface.bmp', cv2.IMREAD_GRAYSCALE)
lighthouse_img = cv2.imread('./images/lighthouse.bmp', cv2.IMREAD_GRAYSCALE)

# Υπολογισμός συχνοτήτων
bridge_freq = calculate_frequencies(bridge_img)
girlface_freq = calculate_frequencies(girlface_img)
lighthouse_freq = calculate_frequencies(lighthouse_img)

# Δημιουργία δέντρου Huffman
bridge_tree = build_huffman_tree(bridge_freq)
girlface_tree = build_huffman_tree(girlface_freq)
lighthouse_tree = build_huffman_tree(lighthouse_freq)

# Δημιουργία των κωδίκων Huffman
bridge_codes = create_huffman_codes(bridge_tree)
girlface_codes = create_huffman_codes(girlface_tree)
lighthouse_codes = create_huffman_codes(lighthouse_tree)

# Κωδικοποίηση των εικόνων
bridge_encoded = encode_image(bridge_img, bridge_codes)
girlface_encoded = encode_image(girlface_img, girlface_codes)
lighthouse_encoded = encode_image(lighthouse_img, lighthouse_codes)

# Αποθήκευση της κωδικοποιημένης εικόνας
save_encoded_image(bridge_encoded, 'bridge_encoded.bin')
save_encoded_image(girlface_encoded, 'girlface_encoded.bin')
save_encoded_image(lighthouse_encoded, 'lighthouse_encoded.bin')

# Αποθήκευση του πίνακα κωδίκων Huffman
save_huffman_codes(bridge_codes, 'bridge_codes.pkl')
save_huffman_codes(girlface_codes, 'girlface_codes.pkl')
save_huffman_codes(lighthouse_codes, 'lighthouse_codes.pkl')

# Υπολογισμός του μέσου μήκους κωδικής λέξης
bridge_avg_length = calculate_average_code_length(bridge_codes, bridge_freq)
girlface_avg_length = calculate_average_code_length(girlface_codes, girlface_freq)
lighthouse_avg_length = calculate_average_code_length(lighthouse_codes, lighthouse_freq)

# Υπολογισμός της εντροπίας
bridge_entropy = calculate_entropy(bridge_freq)
girlface_entropy = calculate_entropy(girlface_freq)
lighthouse_entropy = calculate_entropy(lighthouse_freq)

# Υπολογισμός του λόγου συμπίεσης
bridge_original_size = bridge_img.size * 8  # σε bits
girlface_original_size = girlface_img.size * 8
lighthouse_original_size = lighthouse_img.size * 8

bridge_encoded_size = len(bridge_encoded)  # σε bits
girlface_encoded_size = len(girlface_encoded)
lighthouse_encoded_size = len(lighthouse_encoded)

bridge_compression_ratio = calculate_compression_ratio(bridge_original_size, bridge_encoded_size)
girlface_compression_ratio = calculate_compression_ratio(girlface_original_size, girlface_encoded_size)
lighthouse_compression_ratio = calculate_compression_ratio(lighthouse_original_size, lighthouse_encoded_size)

# Εκτύπωση αποτελεσμάτων
def print_results(name, codes, avg_length, entropy, compression_ratio):
    print(f"Αποτελέσματα για την εικόνα {name}:")
    print("1) Κωδικές λέξεις:")
    for symbol, code in codes.items():
        print(f"  Φωτεινότητα {symbol}: {code}")
    print(f"2) Μέσο μήκος κωδικής λέξης: {avg_length}")
    print(f"3) Εντροπία: {entropy}")
    print(f"4) Λόγος συμπίεσης: {compression_ratio}")
    print()

print_results("bridge.bmp", bridge_codes, bridge_avg_length, bridge_entropy, bridge_compression_ratio)
print_results("girlface.bmp", girlface_codes, girlface_avg_length, girlface_entropy, girlface_compression_ratio)
print_results("lighthouse.bmp", lighthouse_codes, lighthouse_avg_length, lighthouse_entropy, lighthouse_compression_ratio)
