"""Offline review pack and optional independent CTC comparison for word timing."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.sync_audit import environment_inventory, evaluate_annotations, prepare_evidence, run_ctc_audit, verify_manifest, write_report_csv, run_ctc_candidate_comparison, serve_review


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description='生成原音/波形审查包，独立 CTC 辅助对照，以及人工标注误差报告。CTC 不是人工真值。')
    sub = parser.add_subparsers(dest='command', required=True)
    inventory = sub.add_parser('environment', help='只读检查已有运行库，不下载模型')
    inventory.add_argument('--output', type=Path)
    prepare = sub.add_parser('prepare', help='固定分层抽样，提取原音与波形；不执行识别或模型请求')
    prepare.add_argument('--episode', type=Path, required=True)
    prepare.add_argument('--audio', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--minimum', type=int, default=48)
    ctc = sub.add_parser('ctc', help='只对样本局部片段运行独立 wav2vec2 CTC，不重跑整片')
    ctc.add_argument('--manifest', type=Path, required=True)
    ctc.add_argument('--episode', type=Path, help='当前数据包；默认使用审查包快照')
    ctc.add_argument('--cache', type=Path, default=ROOT/'work/sync-audit/model-cache')
    ctc.add_argument('--allow-download', action='store_true', help='显式允许下载约 360.2MiB 的 PyTorch 官方权重到独立缓存')
    ctc.add_argument('--threads', type=int, default=4)
    ctc.add_argument('--limit', type=int, help='只处理前 N 个样本；输出保留未完成人工状态')
    serve = sub.add_parser('serve', help='仅在本机开放审查页及已登记原音，禁止目录列表及其他工程文件')
    serve.add_argument('--folder', type=Path, required=True)
    serve.add_argument('--port', type=int, default=8781)
    compare = sub.add_parser('compare-text', help='同一原音片段比较连续上下文候选的 CTC 模型偏好，不自动改稿')
    compare.add_argument('--audio', type=Path, required=True, help='16kHz 单声道 16-bit PCM WAV，最多 60 秒')
    compare.add_argument('--candidates', type=Path, required=True, help='JSON 数组，或含 candidates 数组的对象；每项 id/text/source，数字或符号须 spoken_form')
    compare.add_argument('--cache', type=Path, default=ROOT/'work/sync-audit/model-cache')
    compare.add_argument('--allow-download', action='store_true')
    compare.add_argument('--threads', type=int, default=4)
    compare.add_argument('--output', type=Path, required=True)
    report = sub.add_parser('report', help='读回标注，计算每词起止误差；不改学习数据')
    report.add_argument('--manifest', type=Path, required=True)
    report.add_argument('--episode', type=Path, required=True, help='校验当前时间轴，拒绝旧标注与新版混用')
    report.add_argument('--annotations', type=Path)
    report.add_argument('--ctc', type=Path)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'environment':
        result = environment_inventory()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == 'prepare':
        result = prepare_evidence(read(args.episode), args.audio, args.output, minimum=args.minimum)
        print(json.dumps({'samples': len(result['samples']), 'strata': result['sampling_counts'], 'human_acceptance': 'not_completed', 'review_page': str(args.output/'review.html')}, ensure_ascii=False))
    elif args.command == 'ctc':
        folder = args.manifest.parent
        package = read(args.episode or folder/'episode-snapshot.json')
        result = run_ctc_audit(package, read(args.manifest), folder, args.cache, allow_download=args.allow_download, threads=args.threads, limit=args.limit)
        manifest = read(args.manifest)
        report_data = evaluate_annotations(manifest, ctc_results=result)
        (folder/'report.json').write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding='utf-8')
        write_report_csv(report_data, folder/'word-boundary-report.csv')
        print(json.dumps({'acoustic_observations': len(result['observations']), 'human_acceptance': report_data['human_acceptance']['status'], 'report': str(folder/'report.json')}, ensure_ascii=False))
    elif args.command == 'serve':
        serve_review(args.folder, port=args.port)
    elif args.command == 'compare-text':
        candidates = read(args.candidates)
        if isinstance(candidates, dict):
            candidates = candidates.get('candidates')
        if not isinstance(candidates, list):
            raise ValueError('候选文件必须提供 candidates 数组。')
        result = run_ctc_candidate_comparison(args.audio, candidates, args.cache, allow_download=args.allow_download, threads=args.threads)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'comparable': result['comparable'], 'ranking_per_frame': result['ranking_per_frame'], 'ranking_per_target_token': result['ranking_per_target_token'], 'human_ground_truth': False, 'output': str(args.output)}, ensure_ascii=False))
    else:
        manifest = read(args.manifest)
        verify_manifest(read(args.episode), manifest)
        result = evaluate_annotations(manifest, read(args.annotations) if args.annotations else None, read(args.ctc) if args.ctc else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        write_report_csv(result, args.output.with_suffix('.csv'))
        print(json.dumps(result['human_acceptance'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('同步审查失败：' + str(error), file=sys.stderr)
        raise SystemExit(1)
