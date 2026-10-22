"""Produce a draft revision; existing course and configuration stay intact."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.acoustic_survey import survey_audio
from app.timing_revision import propose_timing_revision,retime_package
from app.sync_audit import file_sha256

def main():
    parser=argparse.ArgumentParser(description='交叠测量整课词边界并输出修复草案，不覆写原课程，不充当人工验收。')
    parser.add_argument('--episode',type=Path,required=True);parser.add_argument('--audio',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache',type=Path,default=ROOT/'work/sync-audit/model-cache')
    args=parser.parse_args();package=json.loads(args.episode.read_text(encoding='utf-8'))
    if args.output.resolve()==args.episode.parent.resolve():raise ValueError('请使用独立的草案目录。')
    survey=survey_audio(package,args.audio,args.output,args.cache)
    proposal=propose_timing_revision(package['words'],survey,file_sha256(args.audio))
    revised=retime_package(package,proposal)
    for name,value in [('timing-proposal.json',proposal),('episode-draft.json',revised)]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'已输出 {len(proposal["patches"])} 个实测边界修复草案；旧课程保持原样，人工同步验收尚未执行。')

if __name__=='__main__':main()
