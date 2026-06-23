import re

with open('hacker_de_teclado.py', 'r') as f:
    content = f.read()

content = re.sub(
    r'min_dist_sec=0.15',
    'min_dist_sec=0.3',
    content
)

with open('hacker_de_teclado.py', 'w') as f:
    f.write(content)
