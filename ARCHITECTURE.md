# v0.4 实施约定

- 所有实施只修改 universal-studio。旧工程和旧计划任务保持原状。
- 原素材 -> 归一媒体 -> ASR/对齐 -> version 4 EpisodePackage -> 教学/布局 -> 学习页/按需视频。
- 输入 spec: media_path?, url?, title?, source_kind?, source_url?, subtitle_path?, transcript?, audio_stream_index?, start_ms?=0, end_ms?, render_mode?=auto|page|video。时间统一为整数毫秒。
- `app.media.prepare_media(path, folder, audio_stream_index=None, start_ms=0, end_ms=None, progress=None)` 返回 audio_path(真实MP3), video_path(可选H264/AAC MP4), cover_path, duration_s, source_offset_ms, fingerprint, media_info；保留画面比例，统一起点不改变速度。`probe_media(path)` 返回原始 ffprobe 字典。
- `app.sources.inspect_source(url)` 返回 title,duration_s,source_kind,source_url,entries? 等可序列化信息。`acquire_source(url, folder, progress=None)` 返回 media_path, title, source_kind, source_url, subtitle_path?, transcript?, cover_path?。BBC可只有音频；B站不能走yt-dlp。
- `app.pipeline.create_media_episode(spec, job_folder, progress)` 返回package；沿用 ReviewRequired。`progress(stage,pct,message)` 可抛出 `JobPaused`（由server定义/处理即可）；长步骤在片段边界响应暂停。
- `create_bbc_episode` 保留。通用任务经统一 spec 创建；`continue_after_review` 保持旧签名，可增加可选 chapter_id。
- 数据包保留现有 episode/sentences/words/cards/layout；增加 source={kind,name,url,transcript_kind}, media={video_file?,audio_file,duration_s,source_offset_ms,audio_stream_index?}, chapters=[{id,title,start_ms,end_ms,status}], artifacts={study:pending|ready,video:pending|ready|not_requested|failed}。version=4；兼容v3。source_similarity允许null。
- episode.id根据素材内容/音轨/裁剪计算，不能依赖临时路径。字幕/模型/算法版本用于processing缓存指纹。句词卡引用一致。
- 播放时钟唯一：video.currentTime或audio.currentTime；切换布局不能挂载第二个有声播放器。字幕、词义、卡片只从时间推导。
- 后端API：GET /api/status,/api/jobs,/api/episodes,/api/episode?id；POST /api/settings；POST /api/sources/inspect {url}；POST /api/jobs {urls?,specs?}；POST /api/uploads {name,size}=>{upload_id,chunk_size,received_bytes}；PUT /api/uploads/<id> 原始块 + X-Chunk-Offset；GET /api/uploads/<id>；POST /api/uploads/<id>/complete {title?,subtitle_upload_id?,...spec}=>{job_ids}；完成字幕上传用 role=subtitle 不创建job；POST /api/jobs/<id>/pause|resume|retry|continue|export，export {start_ms?,end_ms?}。
- 后端兼容旧 /api/episodes 和 /api/import；UI用新上传入口。job返回id,episode_id,title,status,stage,progress,message,error,package_path,study_url,video_url；input_json不向前端暴露本机绝对路径。
- 独立资料目录默认为项目 work/profile，可由 ENGLISH_VIDEO_DATA_DIR 设置；不自动发现旧资料目录。不安装每日任务。端口8775。
- 发布目录只包含源码/说明/合成演示，排除 work,outputs真实素材,密钥,数据库,日志,个人路径；MIT覆盖自己的源码，第三方媒体不随源码发布。开源仓库由root发布。

文件所有权：root= pipeline/alignment/providers/storage/renderer/study_page/package/install/docs/发布；media agent=app/media.py,app/sources.py,app/subtitle_input.py及对应tests；server agent=app/server.py及对应tests；frontend agent=全部src和scripts/*.mjs及前端测试。公共约定变化需回报root。
