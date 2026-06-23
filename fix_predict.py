import re

with open('hacker_de_teclado.py', 'r') as f:
    content = f.read()

# Replace prediction logic to pad y if end > len(y) instead of continue
old = """            if end > len(y):
                continue
            chunk = y[int(start) : int(end)]"""

new = """            if end > len(y):
                chunk = np.pad(y[int(start):], (0, end - len(y)))
            else:
                chunk = y[int(start) : int(end)]"""

content = content.replace(old, new)

# Also import numpy in predict_audio just in case it's not
with open('hacker_de_teclado.py', 'w') as f:
    f.write(content)
