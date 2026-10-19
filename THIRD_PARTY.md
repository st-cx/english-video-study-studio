# 第三方组件与素材

本仓库作者原创源码的授权声明见 [LICENSE](LICENSE)。它不替代依赖、模型、字体、编解码器或媒体本身的许可。安装、修改或再分发第三方组件时，应保留其版权与许可文件，并按所用版本的上游条款处理。

以下为2026-10-19实际安装环境核对的主要组件，供识别许可边界。Python版本是验证环境快照，`requirements.txt`中的范围允许安装其他兼容版本；前端版本记录在 `package-lock.json`。表格不代替所有间接依赖的完整许可清单。

## 界面与视频制作

| 组件 | 本轮版本 | 上游许可 |
|---|---|---|
| React、ReactDOM | 19.1.1 | MIT；构建页面包含相关代码，保留声明见本页末尾 |
| Vite、React插件 | 7.1.5、5.0.4 | MIT |
| TypeScript | 5.9.2 | Apache-2.0 |
| esbuild | 0.28.1 | MIT |
| Remotion及本项目使用的 `@remotion/*` | 4.0.536 | **Remotion自定义许可**，分Free/Company条件，不能按MIT依赖处理。使用前查看[上游许可](https://www.remotion.dev/license)及安装包的 `LICENSE.md`，尤其核对组织使用与再分发条件 |

## Python与外部工具

| 组件 | 本轮版本 | 上游许可 |
|---|---|---|
| beautifulsoup4 | 4.15.0 | MIT |
| faster-whisper | 1.2.1 | MIT |
| stable-ts | 2.19.1 | MIT |
| CTranslate2 | 4.8.3 | MIT |
| PyAV | 18.1.0 | BSD-3-Clause；其使用的FFmpeg另有许可 |
| PyMuPDF | 1.28.2 | **GNU AGPLv3或Artifex商业许可双许可**，用于PDF文稿读取。不是MIT依赖；参见[上游许可说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) |
| requests | 2.34.2 | Apache-2.0 |
| opencc-python-reimplemented | 0.1.7 | Apache许可，见所安装发行包的许可文件 |
| httpx | 0.28.1 | BSD-3-Clause |
| yt-dlp | 2026.8.19 | Unlicense；本项目通过Python包获取公开YouTube信息与流，某些独立发行文件可能另含第三方组件 |
| bilibili-api-python | 17.4.2 | **GPL-3.0-or-later**。B站适配器直接使用此库的视频接口，不通过yt-dlp，不读取浏览器Cookie |
| FFmpeg、ffprobe | 本机外部安装 | 具体许可取决于所用构建与编解码器，可能涉及LGPL、GPL及其他条件。应以该发行版为准，参见[FFmpeg许可说明](https://ffmpeg.org/legal.html) |

仓库与源码包不内置Python/Node依赖环境、FFmpeg二进制或模型权重。依赖在用户运行安装入口时另行取得。这不意味着组合使用、修改或将其打包再分发时可以忽略各自许可。

Whisper相关模型第一次使用时可能另行下载，应核对实际模型卡和上游许可；不能用本项目的MIT声明为任何模型重新授权。

## 字体、验证素材与原创演示

- 字体由用户本机提供；不随源码分发Arial、微软雅黑或其他商业字体文件。导出画面的字体使用条件遵循该字体自身条款。
- BBC、TED、MIT、YouTube与B站等真实验证素材仅用于本机测试，不进入源码、演示或发布包。公开可下载不等于获得二次分发或投稿许可。
- `demo/`包含项目自编英文脚本、程序生成测试图、Windows英语系统语音生成的原声及其学习包，详见 [演示说明](demo/README.md)。系统语音软件和语音包本身不分发。重新生成时使用者应遵循所选系统语音的适用条款。
- 本项目没有自动投稿功能。导入和使用素材前，请确认自己拥有适用的学习、处理及传播权限。

## React与ReactDOM保留声明

以下声明来自实际安装包，适用于分发页面中包含的React相关代码：

```text
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
