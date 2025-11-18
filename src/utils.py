
import numpy as np

def features_from_smiles(s):
    return [
        len(s),
        sum(1 for c in s if c.isupper()),
        sum(1 for c in s if c.isdigit()),
        s.count('C')
    ]
