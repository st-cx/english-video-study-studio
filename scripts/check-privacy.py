"""Check exactly the staged/tracked release files; never scan private runtime data.

Exit nonzero on suspicious content. Report filenames and rule names only, never
the matching credential or personal text. Can run before or after a Git commit.
"""
from __future__ import annotations
import argparse,json,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'work','outputs','node_modules','.venv','__pycache__','.teaching-cache','private-backups'}
PRIVATE_NAMES={'settings.json','studio.sqlite3','render-props.json','.env','deepseek.key.dpapi'}
PATTERNS={
    'private_windows_profile':re.compile(r'[A-Z]:[\\/]+(?:Users|用户)[\\/]+(?!Public(?:[\\/]|$)|<|YOUR|your|example|test)[^\\/\s"\']+',re.I),
    'private_workspace':re.compile(r'[A-Z]:[\\/]+项目[\\/]+codex',re.I),
    'provider_token':re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'),
    'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

def check(paths):
    issues=[]
    for relative in paths:
        p=Path(relative);name=p.name.lower()
        if any(part.lower() in FORBIDDEN for part in p.parts) or name in PRIVATE_NAMES or name.startswith('.env.') or p.suffix.lower() in {'.dpapi','.sqlite3','.log','.pem','.key'}:
            issues.append({'file':p.as_posix(),'rule':'private_file'});continue
        path=ROOT/p
        if not path.is_file():continue
        try:text=path.read_text(encoding='utf-8')
        except (UnicodeError,OSError):continue
        # Scanner rules themselves intentionally contain pattern fragments.
        if p.as_posix()=='scripts/check-privacy.py':continue
        for rule,pattern in PATTERNS.items():
            if pattern.search(text):issues.append({'file':p.as_posix(),'rule':rule})
    return issues

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--staged',action='store_true');args=parser.parse_args()
    command=['git','diff','--cached','--name-only','-z','--diff-filter=ACMR'] if args.staged else ['git','ls-files','-z']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,check=True)
    paths=[p for p in result.stdout.decode('utf-8').split('\0') if p]
    issues=check(paths)
    print(json.dumps({'status':'failed' if issues else 'passed','files_checked':len(paths),'issues':issues},ensure_ascii=False))
    return 1 if issues else 0

if __name__=='__main__':raise SystemExit(main())
