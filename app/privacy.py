"""Keep runtime metadata and source credentials out of portable learning data."""
from __future__ import annotations
import copy
import re
from urllib.parse import parse_qsl,urlencode,urlsplit,urlunsplit


def public_source_url(value: str) -> str:
    """Keep a public source identity, never reusable URL credentials.

    This is a display/export transform only. Acquisition keeps its original
    URL in the private input spec/cache, so signed media still downloads.
    """
    if not isinstance(value,str):return ''
    value=value.strip()
    try:
        parsed=urlsplit(value)
        if parsed.scheme.lower() not in {'http','https'}:
            return value if not parsed.scheme and not parsed.netloc and not any(c in value for c in ('?','#','\\')) else ''
        if not parsed.hostname:return ''
        hostname=parsed.hostname
        netloc=('['+hostname+']') if ':' in hostname else hostname
        if parsed.port is not None:netloc+=':'+str(parsed.port)
        def sensitive(name):
            compact=re.sub(r'[^a-z0-9]','',name.casefold())
            return (name.casefold().startswith(('x-amz-','x-goog-')) or
                    any(part in compact for part in ('token','signature','secret','credential')) or
                    compact in {'key','apikey','auth','authorization','authkey','oauth','oauth2',
                                'password','passwd','pwd','session','sessionid','ticket','policy',
                                'expires','expiry','expiration','keypairid','googleaccessid',
                                'accesskey','awsaccesskeyid','ossaccesskeyid','xapikey','authsig',
                                'sign','sig','hmac','jwt','hdnea','hdnts','wstime'})
        query=urlencode([(k,v) for k,v in parse_qsl(parsed.query,keep_blank_values=True) if not sensitive(k)])
        return urlunsplit((parsed.scheme.lower(),netloc,parsed.path,query,''))
    except (ValueError,UnicodeError):
        return ''


def public_package(package:dict)->dict:
    result=copy.deepcopy(package)
    result.pop('processing',None)
    def clean(value):
        if isinstance(value,dict):
            return {k:(public_source_url(v) if k in {'url','source_url','audio_url','transcript_url'} else clean(v))
                    for k,v in value.items() if not (k.endswith('_path') or k in {'input_json','api_key','api_key_override','base_url','model','key','_revision','timing_evidence','original_zero_diagnostics'})}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return clean(result)
