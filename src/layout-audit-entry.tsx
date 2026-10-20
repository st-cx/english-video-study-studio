import {Children, cloneElement, isValidElement, useEffect, useMemo, useRef, useState} from 'react';
import type {ReactNode} from 'react';
import {cancelRender, Composition, continueRender, delayRender, registerRoot, staticFile} from 'remotion';
import {LayoutGallery} from './LayoutGallery';
import {ResponsiveScene} from './LessonScene';
import type {EpisodePackage} from './shared';

type AuditProps={episode:EpisodePackage};
type Violation={time_ms:number;kind:string;message:string;selector:string;measured?:Record<string,number>};
const REPORT_PREFIX='ENGLISH_LAYOUT_AUDIT:';
const nextFrame=()=>new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));
function clock(ms:number){const s=Math.max(0,ms)/1000;return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(3).padStart(6,'0')}`;}

function LayoutAudit({episode}:AuditProps){
  const root=useRef<HTMLDivElement>(null);
  const [handle]=useState(()=>delayRender('正在检查全部字幕页与讲解卡排版',{timeoutInMilliseconds:90000}));
  const gallery=useMemo(()=>{
    const coverSrc=episode.episode.cover?staticFile(episode.episode.cover):'';
    // Keep Gallery's exact case selection and markup, adapting its relative
    // offline image URL to Remotion's asset URL (including the no-cover case).
    const adapt=(node:ReactNode):ReactNode=>{
      if(!isValidElement<{children?:ReactNode;coverSrc?:string}>(node))return node;
      if(node.type===ResponsiveScene)return cloneElement(node,{coverSrc});
      return node.props.children?cloneElement(node,{},Children.map(node.props.children,adapt)):node;
    };
    return adapt(LayoutGallery({episode}));
  },[episode]);
  useEffect(()=>{
    let disposed=false;
    async function check(){
      await document.fonts.ready;
      const container=root.current;
      if(!container)throw Error('布局检查未找到场景容器。');
      await Promise.all(Array.from(container.querySelectorAll('img')).map(async img=>{
        await img.decode();
        if(!img.naturalWidth)throw Error('布局检查无法读取节目封面。');
      }));
      // ResponsiveScene receives its final width through ResizeObserver.
      await nextFrame();await nextFrame();
      if(disposed)return;
      const cases=Array.from(container.querySelectorAll<HTMLElement>('.layout-case'));
      if(!cases.length)throw Error('布局检查没有生成任何字幕或讲解场景。');
      const violations:Violation[]=[];
      let checks=0;
      for(const section of cases){
        const time_ms=Number(section.dataset.checkMs||0);
        const add=(kind:string,message:string,selector:string,measured?:Record<string,number>)=>violations.push({time_ms,kind,message,selector,measured});
        const canvas=section.querySelector<HTMLElement>('.scene-canvas');
        const english=section.querySelector<HTMLElement>('.scene-english');
        if(!canvas||!english){add('missing','字幕画面结构缺失','.scene-canvas / .scene-english');continue;}
        for(const [selector,label] of [['.scene-chinese','中文译文'],['.scene-context','当前词义'],['.scene-definition','讲解释义']] as const){
          for(const node of section.querySelectorAll<HTMLElement>(selector)){
            checks++;
            if(node.scrollHeight>node.clientHeight+2)add('vertical-overflow',`${label}超出显示高度`,selector,{scrollHeight:node.scrollHeight,clientHeight:node.clientHeight});
            if(node.scrollWidth>node.clientWidth+2)add('horizontal-overflow',`${label}超出显示宽度`,selector,{scrollWidth:node.scrollWidth,clientWidth:node.clientWidth});
          }
        }
        for(const line of section.querySelectorAll<HTMLElement>('.scene-caption-line')){
          checks++;
          if(line.scrollWidth>line.clientWidth+2)add('caption-overflow','英文字幕超出固定行宽','.scene-caption-line',{scrollWidth:line.scrollWidth,clientWidth:line.clientWidth});
        }
        const englishTop=(section.querySelector<HTMLElement>('.scene-subtitle-area')||english).getBoundingClientRect().top;
        const tolerance=canvas.getBoundingClientRect().width/1920*2;
        for(const [selector,label] of [['.scene-context','当前词义'],['.scene-focus-card','讲解卡'],['.scene-intro','主题卡']] as const){
          for(const node of section.querySelectorAll<HTMLElement>(selector)){
            checks++;
            const bottom=node.getBoundingClientRect().bottom;
            if(bottom>englishTop-tolerance)add('subtitle-overlap',`${label}与字幕及其来源行区域重叠`,selector,{bottom:Math.round(bottom*100)/100,englishTop:Math.round(englishTop*100)/100});
          }
        }
      }
      console.log(REPORT_PREFIX+JSON.stringify({version:1,status:violations.length?'failed':'passed',cases:cases.length,checks,violations}));
      if(violations.length){
        const details=violations.slice(0,6).map(v=>`${clock(v.time_ms)} ${v.message}`).join('；');
        cancelRender(new Error(`画面检查未通过，共 ${violations.length} 处需要检查：${details}。`));
        return;
      }
      continueRender(handle);
    }
    check().catch(error=>{if(!disposed)cancelRender(error instanceof Error?error:new Error(String(error)));});
    return()=>{disposed=true;};
  },[episode,gallery,handle]);
  return <div ref={root}>{gallery}</div>;
}

const emptyEpisode:EpisodePackage={version:3,episode:{id:'layout-audit',title:'画面检查',title_zh:'画面检查',source_url:'',audio_url:'',duration_s:1,source_date:'',cover:'',audio_file:''},sentences:[],words:[],cards:[],quality:{needs_review:false,flags:[],source_similarity:1},review:{recognized_transcript:'',official_transcript:'',approved:false},render:{}};
registerRoot(()=> <Composition id="LayoutAudit" component={LayoutAudit} width={1920} height={1080} fps={30} durationInFrames={1} defaultProps={{episode:emptyEpisode}}/>);
