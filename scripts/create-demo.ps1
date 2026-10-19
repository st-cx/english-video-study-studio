param([string]$OutputDirectory = "work/public-demo-input")
$ErrorActionPreference = 'Stop'
$demoRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDirectory))
New-Item -ItemType Directory -Force -Path $demoRoot | Out-Null
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$english = $voice.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1
if (-not $english) { throw '请安装一个Windows英语语音包，再生成演示。' }
$voice.SelectVoice($english.VoiceInfo.Name)
$voice.Rate = -1
$text = 'Welcome to your English study studio. You can learn from short clips, interviews, and long courses. Listen to one sentence, notice useful expressions, and then play it again. Small steps can make a real difference.'
$audioFile = Join-Path $demoRoot 'speech.wav'
$voice.SetOutputToWaveFile($audioFile)
$voice.Speak($text)
$voice.Dispose()
[System.IO.File]::WriteAllText((Join-Path $demoRoot 'script.txt'), $text, [System.Text.UTF8Encoding]::new($false))
$videoFile = Join-Path $demoRoot 'original-demo.mp4'
& ffmpeg -v error -f lavfi -i 'testsrc2=size=960x540:rate=30' -i $audioFile -map 0:v:0 -map 1:a:0 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac -shortest -map_metadata -1 -y $videoFile
if ($LASTEXITCODE -ne 0) { throw '演示视频生成失败。' }
Write-Output $videoFile
