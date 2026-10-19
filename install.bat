@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHON_LAUNCHER="
if exist ".venv\Scripts\python.exe" goto dependencies
where py >nul 2>nul
if not errorlevel 1 (
  py -3.13 -m venv .venv >nul 2>nul
  if not errorlevel 1 set "PYTHON_LAUNCHER=py -3.13"
  if not defined PYTHON_LAUNCHER py -3.12 -m venv .venv >nul 2>nul
  if not defined PYTHON_LAUNCHER if exist ".venv\Scripts\python.exe" set "PYTHON_LAUNCHER=py -3.12"
  if not defined PYTHON_LAUNCHER py -3.11 -m venv .venv >nul 2>nul
  if not defined PYTHON_LAUNCHER if exist ".venv\Scripts\python.exe" set "PYTHON_LAUNCHER=py -3.11"
)
if not defined PYTHON_LAUNCHER (
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
  if errorlevel 1 (
    echo 请先安装 Python 3.11 至 3.13，再重新运行安装。
    pause
    exit /b 1
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo 未能创建独立 Python 环境，请检查 Python 安装。
  pause
  exit /b 1
)
:dependencies
".venv\Scripts\python.exe" -c "import sys;sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) else 1)"
if errorlevel 1 (
  echo 本工具需要 Python 3.11 至 3.13。请保留现有数据后使用兼容环境。
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo 找不到 FFmpeg。请先安装 FFmpeg 并加入 PATH。
  pause
  exit /b 1
)
where ffprobe >nul 2>nul
if errorlevel 1 (
  echo 找不到 ffprobe，请安装完整的 FFmpeg 工具包并加入 PATH。
  pause
  exit /b 1
)
echo 正在安装本地语音与素材处理工具……
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Python 依赖安装未完成，请检查网络后重新运行。
  pause
  exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
  echo 请先安装 Node.js 22.12 或更新版本，再重新运行安装。
  pause
  exit /b 1
)
node -e "let [v,m]=process.versions.node.split('.').map(Number);process.exit((v===20&&m>=19)||(v===22&&m>=12)||v>22?0:1)"
if errorlevel 1 (
  echo 当前 Node.js 版本不兼容，请升级到 22.12 或更新版本。
  pause
  exit /b 1
)
echo 正在安装本地视频制作与界面工具……
call npm ci
if errorlevel 1 (
  echo 视频工具安装未完成，请检查网络后重新运行。
  pause
  exit /b 1
)
echo 正在准备中文界面……
call npm run build
if errorlevel 1 (
  pause
  exit /b 1
)
echo 安装完成，请双击 start.bat 打开学习工作室。
pause
