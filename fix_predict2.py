import re

with open('hacker_de_teclado.py', 'r') as f:
    content = f.read()

content = re.sub(
    r'model_path="keystroke_model_best.pth"',
    'model_path="keystroke_model.pth"',
    content
)

with open('hacker_de_teclado.py', 'w') as f:
    f.write(content)
