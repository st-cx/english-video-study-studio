from __future__ import annotations
import argparse
from app.server import serve

def main() -> int:
    parser=argparse.ArgumentParser(description='英语视频精读精听工作室')
    parser.add_argument('--port',type=int,default=8775)
    parser.add_argument('--open',action='store_true',help='打开本地学习工作室')
    args=parser.parse_args()
    serve(args.port,args.open)
    return 0

if __name__=='__main__': raise SystemExit(main())
