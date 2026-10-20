"""Package tracked source/build/demo only; exclude all runtime and user media."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import importlib.util
spec=importlib.util.spec_from_file_location('privacy_check',ROOT/'scripts/check-privacy.py');privacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(privacy)

def main():
    for args in (['diff','--quiet'],['diff','--cached','--quiet']):
        if subprocess.run(['git',*args],cwd=ROOT).returncode:
            raise RuntimeError('请先提交所有源码修改，再打包发布，确保发布包可追溯。')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=[s for s in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0') if s]
    issues=privacy.check(names)
    if issues:raise RuntimeError('隐私检查失败：'+json.dumps(issues,ensure_ascii=False))
    files=[ROOT/name for name in names]
    files.extend(p for p in (ROOT/'dist').rglob('*') if p.is_file() and not p.name.endswith('.map'))
    # Ignore no runtime tree even if accidentally added to the list later.
    files=sorted(set(files))
    issues=privacy.check([p.relative_to(ROOT).as_posix() for p in files])
    if issues:raise RuntimeError('发布文件隐私检查失败：'+json.dumps(issues,ensure_ascii=False))
    for p in files:
        if any(x.lower() in privacy.FORBIDDEN for x in p.relative_to(ROOT).parts):raise RuntimeError('发布清单包含私有资料目录。')
    entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    version=json.loads((ROOT/'package.json').read_text(encoding='utf-8'))['version']
    if not __import__('re').fullmatch(r'\d+\.\d+\.\d+',version):raise RuntimeError('版本号格式无效。')
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    final=out/f'english-video-study-studio-v{version}.zip';temporary=final.with_suffix('.zip.part')
    with ZipFile(temporary,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('RELEASE-MANIFEST.json',json.dumps({'version':version,'commit':commit,'files':entries},ensure_ascii=False,indent=2))
    with ZipFile(temporary) as z:
        assert z.testzip() is None
        for e in entries:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
    os.replace(temporary,final)
    metadata={'file':final.name,'commit':commit,'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'bytes':final.stat().st_size,'entries':len(entries)+1}
    final.with_suffix('.sha256.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False))

if __name__=='__main__':main()
