"""Build an evidence-bound review draft without overwriting a live lesson."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.review_reconcile import reconcile_reviews
from app.storage import save_episode


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description='根据已有原音证据复核待检查问题，输出独立草案；不改配置，不代表人工精度验收。')
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', action='append', required=True, help='本机已存在的无提示识别模型目录，可重复')
    parser.add_argument('--observations', type=Path, action='append', default=[], help='已测无提示观察JSON数组，可重复')
    parser.add_argument('--ctc', type=Path, action='append', default=[], help='完整上下文CTC候选比较JSON，可重复')
    parser.add_argument('--survey', type=Path, action='append', default=[], help='交叠词边界测量JSON，可重复')
    args = parser.parse_args()
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
    result = reconcile_reviews(read(args.episode), args.audio, model_paths=args.model_path,
                               observations=observations, ctc_observations=[read(p) for p in args.ctc],
                               boundary_surveys=[read(p) for p in args.survey])
    output.mkdir(parents=True, exist_ok=True)
    save_episode(result['episode'], output, 'episode-draft.json')
    save_episode({k: v for k, v in result.items() if k != 'episode'}, output, 'decisions.json')
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
