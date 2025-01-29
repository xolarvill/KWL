import numpy as np

def compare_vectors(v1, v2):
    if np.allclose(v1, v2):
        return "相同"
    else:
        differences = []
        for i in range(len(v1)):
            if not np.isclose(v1[i], v2[i]):
                differences.append(f"参数{i+1}不同: {v1[i]} vs {v2[i]}")
        return "\n".join(differences)