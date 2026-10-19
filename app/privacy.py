"""Keep runtime-only metadata out of portable learning pages."""
from __future__ import annotations
import copy

def public_package(package:dict)->dict:
    result=copy.deepcopy(package)
    result.pop('processing',None)
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items() if not (k.endswith('_path') or k in {'input_json','api_key','api_key_override','base_url','model','key','_revision'})}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return clean(result)
