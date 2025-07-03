# function.py

import pandas as pd
from collections import Counter
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
import re
import regex  # hoặc import re nếu bạn không dùng Unicode nâng cao
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# =============================
# Danh sách từ tích cực & tiêu cực
# =============================

positive_words = [
    "good", "great", "excellent", "efficient", "supportive", "friendly", "creative",
    "enthusiastic", "passionate", "dedicated", "professional", "reliable", "fun",
    "motivated", "inspiring", "productive", "collaborative", "trustworthy", "cheerful",
    "positive", "comfortable", "encouraging", "flexible", "respectful", "engaging",
    "helpful", "innovative", "stable", "welcoming", "rewarding"
]


negative_words = [
    "bad", "poor", "inefficient", "stressful", "toxic", "unfriendly", "boring",
    "unmotivated", "disorganized", "unprofessional", "rude", "inflexible", "overworked",
    "underpaid", "frustrating", "micromanaged", "unfair", "slow", "confusing",
    "demanding", "negative", "pressured", "annoying", "hostile", "exhausting",
    "chaotic", "inconsistent", "isolated", "unappreciated", "low"
]

rating_columns = [
    'Salary & benefits',
    'Training & learning',
    'Management cares about me',
    'Culture & fun',
    'Office & workspace'
]

# =============================
# Hàm xử lý
# =============================

def count_pos_neg_words(text, pos_words=positive_words, neg_words=negative_words):
    """
    Đếm số từ tích cực và tiêu cực trong văn bản

    Parameters:
        text (str): đoạn văn bản đã xử lý
        pos_words (list): danh sách từ tích cực
        neg_words (list): danh sách từ tiêu cực

    Returns:
        pos_count (int), neg_count (int)
    """
    text = text.replace("_", " ").lower()
    tokens = text.split()
    counter = Counter(tokens)
    pos_count = sum(counter[w] for w in pos_words if w in counter)
    neg_count = sum(counter[w] for w in neg_words if w in counter)
    return pos_count, neg_count


def classify_sentiment(row):
    """
    Gán nhãn cảm xúc dựa vào đánh giá và số từ cảm xúc

    Parameters:
        row (pd.Series): 1 dòng dữ liệu chứa ratings + pos_count + neg_count

    Returns:
        int: 0 = tiêu cực, 1 = trung tính, 2 = tích cực
    """
    ratings = row[rating_columns].values
    pos_count = row['pos_count']
    neg_count = row['neg_count']

    if all(r == 5 for r in ratings):
        return 2
    elif all(r >= 4 for r in ratings) and (pos_count > neg_count):
        return 2
    elif any(r <= 2 for r in ratings) and (ratings.mean() <= 3 or pos_count <= neg_count):
        return 0
    else:
        return 1



# ======================================
# Hàm load dictionary từ file txt
# ======================================

# def load_dictionaries():
#     """
#     Load tất cả dictionary từ các file txt:
#     - emoji_dict
#     - teen_dict
#     - english_dict
#     - wrong_lst
#     - stopwords_lst

#     Trả về: tuple gồm 5 đối tượng
#     """
#     # Emoji
#     emoji_dict = {}
#     with open("emojicon.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 key, val = parts
#                 emoji_dict[key] = val

#     # Teencode
#     teen_dict = {}
#     with open("teencode.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 key, val = parts
#                 teen_dict[key] = val

#     # # English-Vietnamese
#     # english_dict = {}
#     # with open("english-vnmese.txt", "r", encoding="utf-8") as f:
#     #     for line in f:
#     #         parts = line.strip().split('\t')
#     #         if len(parts) == 2:
#     #             key, val = parts
#     #             english_dict[key] = val

#     # Từ sai
#     wrong_lst = []
#     with open("wrong-word.txt", "r", encoding="utf-8") as f:
#         wrong_lst = [line.strip() for line in f if line.strip()]

#     # Stopwords
#     stopwords_lst = []
#     with open("vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
#         stopwords_lst = [line.strip() for line in f if line.strip()]

#     return emoji_dict, teen_dict, wrong_lst, english_dict, stopwords_lst


#=================================================================
# Hàm rút gọn địa điểm
# Hàm rút gọn địa điểm
def extract_city(location_str):
    if pd.isna(location_str):
        return "Unknown"
    location_str = location_str.lower()
    if "hà nội" in location_str or "ha noi" in location_str:
        return "Hà Nội"
    elif "hồ chí minh" in location_str or "ho chi minh" in location_str or "hcm" in location_str:
        return "Hồ Chí Minh"
    elif "đà nẵng" in location_str or "da nang" in location_str:
        return "Đà Nẵng"
    elif "cần thơ" in location_str or "can tho" in location_str:
        return "Cần Thơ"
    else:
        return "Khác"
    
#=================================================================
# Chuẩn hóa về tiếng Anh
def fast_translate(text):
    if pd.isnull(text) or text.strip() == "":
        return ""
    try:
        return GoogleTranslator(source='vi', target='en').translate(text)
    except:
        return ""
    
#=================================================================
# 
def process_text(text):
    document = text.lower()
    document = document.replace("’", '')
    document = regex.sub(r'\.+', ".", document)
    new_sentence = ''
    for sentence in sent_tokenize(document):
        # # Convert emoji
        # sentence = ''.join(emoji_dict[word] + ' ' if word in emoji_dict else word for word in list(sentence))
        # # Convert teencode
        # sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        # # Remove wrong words
        # sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        # Remove English stopwords
        stop_words = set(stopwords.words('english'))
        sentence = ' '.join(word for word in sentence.split() if word not in stop_words)
        new_sentence += sentence + '. '
    document = new_sentence
    document = regex.sub(r'\s+', ' ', document).strip()
    return document    