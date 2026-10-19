# 公开来源验证记录

记录日期：2026-10-19。页面预检、媒体下载、归一和完整教学制作分别记录；一个步骤成功不代表整个平台或完整制作已经通过。

| 平台与公开视频 | 真实测试状态 |
|---|---|
| [BBC：How reading shapes your brain](https://www.bbc.co.uk/learningenglish/features/6-minute-english_2026/ep-260514) | 新版来源模块已实际读取官方页面并提取正确节目标题。此记录不将旧版样片等同于新版获取流程的完整验收 |
| [YouTube：Me at the zoo](https://www.youtube.com/watch?v=jNQXAC9IVRw) | 页面预检、公开画面流/音轨、人工英文 VTT和封面获取成功，媒体实测19.063秒。实际经HTTP上传、识别对齐、待检查修正与继续后完整交付，任务为complete；另经HTTP导出4–8秒片段，实际4秒、1920×1080、30fps、有音轨。对应原声段相关0.9999、测得偏移0ms；该结果不代替人工词边界听辨验收 |
| [YouTube／TED：Try something new for 30 days](https://www.youtube.com/watch?v=JnfBXjWm7hc) | 页面预检、公开音视频、人工英文 VTT和封面获取成功，媒体实测207.4秒；最终函数教学复核6/6批明确通过，60项画面cases零违规；完整MP4实测207.403秒、1920×1080、30fps，开中末均可解码，学习页与视频ready |
| [YouTube／MIT：What is Computation?](https://www.youtube.com/watch?v=nykOeWgQcHM) | 页面预检成功（标示 2586 秒，即 43 分 6 秒）。1080p 画面公开流可传输，但因验证耗时暂停并保留断点；480p 选择返回真实 HTTP 403。未将此入口计为完整下载通过，转用下一行的官方离线下载 |
| [MIT 官方课程页](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/resources/lecture-1-what-is-computation/)列出的[同一课程离线视频](https://archive.org/download/MIT6.0001F16/MIT6_0001F16_Lecture_01_300k.mp4) | 通过新版公开直链模块完整下载并验证，101,137,401 字节，实测 2585.964 秒，480×360 H.264 画面、48kHz 双声道 AAC 原声。官方英文 VTT 已下载并读取出 887 条对白。全片已完成9段声学处理，6,673词/555句；存在实词差异、零时长与重叠，9单元进入待检查，未完成中文教学与MP4，不计为长片闭环通过 |
| [哔哩哔哩：阅读如何塑造你的大脑](https://www.bilibili.com/video/BV1XrL56vEgu/) | 补齐独立环境的 HTTP 后端后，公开元数据和取流成功：标示 358 秒，返回 6 路 DASH 画面和 3 路音频。未重复下载大素材，尚未验证此入口的完整下载与教学制作。最初错误来自缺少 HTTP 请求后端，不能写成平台限制 |

上述测试没有读取浏览器 Cookie 或替用户登录。YouTube 使用 yt-dlp 读取公开信息；哔哩哔哩使用 bili-cli 底层 bilibili_api 视频接口，未使用 yt-dlp。真实访问可能随来源状态、网络、登录要求和公开流有效期变化；失败应保留具体阶段，并允许改用本地导入。
