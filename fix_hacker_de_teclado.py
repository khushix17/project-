import re

with open('hacker_de_teclado.py', 'r') as f:
    content = f.read()

content = re.sub(
    r'onsets = librosa\.onset\.onset_detect\(\s*y=y,\s*sr=sr,\s*backtrack=True,\s*units="samples"\s*\)',
    'onsets = get_filtered_onsets(y, sr)',
    content
)

helper_code = """
def get_filtered_onsets(y, sr, min_dist_sec=0.15):
    import librosa
    raw_onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="samples")
    filtered = []
    min_dist = sr * min_dist_sec
    for o in raw_onsets:
        if not filtered or (o - filtered[-1]) > min_dist:
            filtered.append(o)
    return filtered
"""

content = content.replace("from torch.optim.lr_scheduler import LinearLR\n", "from torch.optim.lr_scheduler import LinearLR\n" + helper_code + "\n")

with open('hacker_de_teclado.py', 'w') as f:
    f.write(content)
