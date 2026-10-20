# v0.5 实施约定

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

## 原声证据与长课修订

- 大模型优先选择本机完整缓存，GPU使用int8_float16；CPU回退必须记录真实模型与计算类型，缓存指纹不得冒称GPU大模型。ASR与stable对齐共享一个模型，完成后释放再进入独立CTC。
- speech_chunks里的所有word、whisper_segment和alignment诊断统一到整片时间；clip局部word_index不用于全片索引算术，争议按文字与实测时段唯一匹配。
- 全片参考稿只比较一次，按实测时段分配争议；跨章cue作为上下文，不能重复计算整句漏词。
- reference_resolution是自动参考门禁：绑定当前原声、模型、争议和候选证据。精确text/time指纹不受ID、中文词义等教学元数据影响；英文或时间变动会使旧证据失效。
- CTC修复只接受双窗稳定且邻词联合兼容的实测跨度；低分、重复和未确认读法保持检查。真实变更保留原始诊断和证据，不通过approved布尔清旗。
- 并行排版使用独立临时文件；可选讲解建议无法定位原句时有界修复后省略并记录。必需词义/翻译补齐失败仍暂停；教学复核格式错误可修复，往返改写仍待检查。
- sync_audit分别记录自动声学对照与人工听辨；程序不能代填已听辨声明。课程原音、运行配置和验收包仅存本机，不进入公开发布。

## v0.5.1 内容目录与列表跟随

- `chapters`保留制作/检查单元。`study_sections={version,status,source_fingerprint,items}`用于内容导航，条目含title、summary、start_ms/end_ms、start_sentence_id、sentence_ids及evidence_sentence_ids；主题可跨制作单元，其显示状态由所覆盖单元推导。
- 模型只选择请求内的短原句编号，程序精确映射回原句ID和实测时间；所有主题连续覆盖整课。英文或句子时段改变会使目录缓存失效，摘要不会更改句词或检查状态。
- 播放器时间先推导当前句/卡，再触发列表内部定位。跟随只在目标或主题改变时工作；不滚动整个文档。滚轮、触摸和列表按键暂停跟随，显式返回当前内容可恢复。
