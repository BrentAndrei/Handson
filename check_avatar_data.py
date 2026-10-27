import os
import re

labels_54 = [
    'A', 'AGAIN', 'B', 'C', 'CARD', 'CASH', 'COIN', 'Confirm',
    'D', 'DISCOUNT', 'E', 'EIGHT', 'F', 'FIVE', 'FOUR', 'G', 'H',
    'HELLO', 'HOW_MANY', 'HOW_MUCH', 'I', 'J', 'K', 'L', 'M', 'N',
    'NINE', 'NO', 'O', 'ONE', 'P', 'PLEASE', 'PROBLEM', 'Q', 'R',
    'RECEIPT', 'S', 'SEVEN', 'SIX', 'Space', 'Speak', 'T', 'TEN',
    'THANK_YOU', 'THREE', 'TWO', 'U', 'V', 'W', 'WAIT', 'WELCOME',
    'X', 'Y', 'YES'
]

for label in labels_54:
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', label)
    filename = f'{sanitized}.json'
    path = f'public/avatar-data/{filename}'
    exists = os.path.exists(path)
    if not exists:
        print(f'MISSING: {label} -> {filename}')
    else:
        print(f'OK: {label} -> {filename}')