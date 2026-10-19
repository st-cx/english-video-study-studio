import type {FC} from 'react';
import {AbsoluteFill,Audio,Composition,Img,OffthreadVideo,staticFile,useCurrentFrame,useVideoConfig} from 'remotion';
import type {EpisodePackage} from './shared';
import {mediaFiles} from './shared';
import {LessonScene} from './LessonScene';

type LessonProps={episode:EpisodePackage;startMs?:number;endMs?:number};
export const Lesson:FC<LessonProps>=({episode,startMs=0})=>{
  const frame=useCurrentFrame(),{fps}=useVideoConfig(),media=mediaFiles(episode);
  const startFrom=Math.round(startMs/1000*fps);
  return <AbsoluteFill><LessonScene episode={episode} timeMs={startMs+frame/fps*1000} coverSrc={episode.episode.cover?staticFile(episode.episode.cover):''} videoSrc={media.video?staticFile(media.video):undefined} videoComponent={OffthreadVideo} videoStartFrame={startFrom} imageComponent={Img}/><Audio src={staticFile(media.audio)} startFrom={startFrom}/></AbsoluteFill>;
};
const placeholder={version:4,episode:{id:'preview',title:'English in context',title_zh:'英语精听',source_url:'',audio_url:'',source_date:'',duration_s:10,audio_file:'audio.mp3',cover:''},sentences:[],words:[],cards:[],quality:{needs_review:false,flags:[],source_similarity:null},review:{recognized_transcript:'',official_transcript:'',approved:false},render:{}} as EpisodePackage;
export const LessonRoot:FC=()=> <Composition id="Lesson" component={Lesson} width={1920} height={1080} fps={30} durationInFrames={300} defaultProps={{episode:placeholder,startMs:0} as LessonProps} calculateMetadata={({props})=>{
  const start=props.startMs||0,end=props.endMs??props.episode.episode.duration_s*1000;
  if(start<0||end<=start||end>props.episode.episode.duration_s*1000+1)throw Error('视频导出范围无效。');
  return {durationInFrames:Math.max(1,Math.ceil((end-start)/1000*30))};
}}/>;
