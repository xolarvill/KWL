from typing import List

def get_prov_code_rank(prov,list:List):
    return list.index(prov) if prov in list else -1