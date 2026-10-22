"""Build an evidence-bound review draft without overwriting a live lesson."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.review_reconcile import (
    propose_disputed_boundary_revisions, rebind_ctc_sources, reconcile_reviews,
)
from app.storage import save_episode
from app.sync_audit import digest, file_sha256
from app.timing_revision import retime_package


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='根据已有原音证据复核待检查问题，输出独立草案；不改配置，不代表人工精度验收。')
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', action='append', required=True, help='本机已存在的无提示识别模型目录，可重复')
    parser.add_argument('--observations', type=Path, action='append', default=[], help='已测无提示观察JSON数组，可重复')
    parser.add_argument('--ctc', type=Path, action='append', default=[], help='完整上下文CTC候选比较JSON，可重复')
    parser.add_argument('--survey', type=Path, action='append', default=[], help='交叠词边界测量JSON，可重复')
    parser.add_argument('--boundary-survey', type=Path, action='append', default=[],
        help='显式启用仅草案的争议词实测改时；需与当前原声及整份词语SHA一致的测量JSON，可重复；不覆写原课')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == args.episode.resolve().parent or output / 'episode-draft.json' == args.episode.resolve():
        parser.error('输出请选择独立草案目录，不能使用实际课程目录。')
    observations = []
    for path in args.observations:
        value = read(path)
        rows = value if isinstance(value, list) else value.get('observations', [])
        if not isinstance(rows, list):
            raise ValueError('无提示观察必须是数组。')
        observations.extend(rows)
    package = read(args.episode)
    ctc_sources = [read(path) for path in args.ctc]
    surveys = [read(path) for path in args.survey]
    timing_proposal = None
    changed_timing_sentence_ids = []
    if args.boundary_survey:
        audio_hash = file_sha256(args.audio)
        word_hash = digest(package.get('words', []))
        requested = [read(path) for path in args.boundary_survey]
        for survey in requested:
            if (not isinstance(survey, dict) or survey.get('audio_sha256') != audio_hash or
                    survey.get('input_timing_sha256') != word_hash):
                raise ValueError('争议边界测量与当前原声或完整词语版本SHA不一致；已停止，原课和草案均未覆写。')
        timing_proposal = propose_disputed_boundary_revisions(package, args.audio,
            model_paths=args.model_path, observations=observations, boundary_surveys=requested)
        if timing_proposal.get('patches'):
            original = package
            package = retime_package(package, timing_proposal)
            ctc_sources = rebind_ctc_sources(original, package, ctc_sources, audio_hash)
            changed = {patch['word_id'] for patch in timing_proposal['patches']}
            changed_timing_sentence_ids = sorted({word['sentence_id'] for word in package['words']
                                                 if word['id'] in changed})
        surveys.extend(requested)
    result = reconcile_reviews(package, args.audio, model_paths=args.model_path,
                               observations=observations, ctc_observations=ctc_sources,
                               boundary_surveys=surveys)
    if timing_proposal is not None:
        result['summary'].update(timing_repairs=len(timing_proposal.get('patches', [])),
            changed_timing_sentence_ids=changed_timing_sentence_ids,
            timing_proposal_file='boundary-proposal.json', human_acceptance='not_completed')
    output.mkdir(parents=True, exist_ok=True)
    if timing_proposal is not None:
        save_episode(timing_proposal, output, 'boundary-proposal.json')
        save_episode(result['summary'], output, 'summary.json')
    save_episode(result['episode'], output, 'episode-draft.json')
    save_episode({k: v for k, v in result.items() if k != 'episode'}, output, 'decisions.json')
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
