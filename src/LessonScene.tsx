import React, {useEffect, useRef, useState} from 'react';
import type {EpisodePackage,TimedWord} from './shared';
import {deriveLessonState, formatClock, cardHeading, sourceName, episodeIndex} from './shared';
import './lesson.css';

type SceneProps={episode:EpisodePackage;timeMs:number;animate?:boolean;coverSrc?:string;videoSrc?:string;videoComponent?:React.ElementType;videoStartFrame?:number;hideChinese?:boolean;imageComponent?:React.ElementType;onWordSelect?:(word:TimedWord)=>void};
function Mark({text,term}:{text:string;term:string}){
  if(!term||!text.toLowerCase().includes(term.toLowerCase()))return <>{text}</>;
  const at=text.toLowerCase().indexOf(term.toLowerCase());
  return <>{text.slice(0,at)}<mark>{text.slice(at,at+term.length)}</mark>{text.slice(at+term.length)}</>;
}
export function LessonScene({episode,timeMs,animate=true,coverSrc,videoSrc,videoComponent:Video='video',videoStartFrame,hideChinese=false,imageComponent:Image='img',onWordSelect}:SceneProps){
  const state=deriveLessonState(episode,timeMs),card=state.card,unit=state.glossUnit;
  const source=coverSrc??episode.episode.cover;
  const cardFade=card&&animate?Math.min(1,Math.max(0,(timeMs-card.start_ms)/180)):1;
  const progress=Math.max(0,Math.min(1,timeMs/(episode.episode.duration_s*1000)));
  const index=episodeIndex(episode),sentenceIndex=index.sentenceIndex.get(state.sentence?.id||'')??-1;
  const outro=timeMs>(episode.sentences.at(-1)?.end_ms??Infinity)+1200;
  const displayCards=index.cards,origin=sourceName(episode);
  return <div className="scene-canvas" data-time-ms={Math.round(timeMs)} data-page={state.captionPage?.id||''}>
    <div className="scene-atmosphere" />
    {source&&<div className="scene-photo-wash" style={{backgroundImage:`url("${source}")`}}/>}
    <div className="scene-vignette" />
    <div className="scene-brand"><b>▶</b><span>英语 · 精听<small>ENGLISH IN CONTEXT</small></span></div>
    <div className={`scene-chinese ${hideChinese?'scene-chinese-hidden':''}`}><span>{hideChinese?'先听原声，再核对译文':outro?'再听一遍，留意刚学过的表达。':state.captionPage?.translation||state.sentence?.translation||episode.episode.title_zh}</span></div>
    <div className="scene-cover-area">
      <div className="scene-cover-frame">{videoSrc?<Video className="scene-source-video" src={videoSrc} startFrom={videoStartFrame} muted/>:source?<Image className="scene-cover-image" src={source} alt={episode.episode.title}/>:<div className="scene-cover-fallback"><small>{origin}</small><strong>{episode.episode.title}</strong></div>}</div>
      <div className="scene-cover-caption"><span className="scene-source-dot"/>{origin}<span>原声 · 双语精读</span></div>
      <div className={`scene-context ${unit?.phrase?'scene-context-phrase':''}`} data-term-id={unit?.id||''}>
        <div className="scene-context-heading"><span>{unit?.phrase?'当前词组':'当前词义'}</span><i>{state.activeWord?'● 正在读':'● 原声精听'}</i></div>
        <div className="scene-context-body"><strong>{unit?.text||(outro?'Listen again':'Listen & notice')}</strong><span>{hideChinese?'':unit?.gloss_zh||''}</span></div>
      </div>
    </div>
    <div className="scene-focus-area">
      {card?<article className="scene-focus-card" style={{opacity:cardFade,transform:`translateY(${(1-cardFade)*7}px)`}} data-card-id={card.id}>
        <div className="scene-focus-top"><span className={`scene-label ${card.type==='grammar'?'scene-label-purple':''}`}>{card.type==='grammar'?'句型拆解':'重点表达'}</span><span className="scene-focus-number">{String((index.cardIndex.get(card.id)??0)+1).padStart(2,'0')} / {String(displayCards.length).padStart(2,'0')}</span></div>
        <h2>{cardHeading(card,hideChinese)}</h2>
        {!hideChinese&&card.show_meaning!==false&&<p className="scene-definition">{card.meaning}</p>}
        {!hideChinese&&(card.type==='grammar'&&card.points?.length?<div className="scene-breakdown">{card.points.slice(0,3).map((p,i)=><div key={i}><span>{p.label}</span><b>{p.text}</b></div>)}</div>:card.structure&&<div className="scene-collocation"><span>{card.type==='grammar'?'结构':'用法'}</span><b>{card.structure}</b></div>)}
        {card.example&&<div className="scene-source-example"><small>原句片段 · IN CONTEXT</small><p><Mark text={card.example} term={card.type==='vocab'?card.title:''}/></p>{!hideChinese&&card.type!=='grammar'&&<span>{card.translation}</span>}</div>}
      </article>:<div className="scene-intro">
        <div className="scene-focus-top"><span className="scene-label">本期主题</span><span className="scene-focus-number">{origin}</span></div>
        <h1>{outro?'A little every day.':episode.episode.title}</h1>{!hideChinese&&<h2>{outro?'今天的精听练习完成了。':episode.episode.title_zh}</h2>}
        <div className="scene-intro-line"/>
        <p>听原声 · 理解语境 · 记住表达</p>
        <div className="scene-preview-terms">{episode.cards.slice(0,3).map(c=><span key={c.id}>{c.type==='grammar'?'句型':c.title}</span>)}</div>
      </div>}
    </div>
    <div className="scene-subtitle-area">
      <div className="scene-subtitle-meta"><span>{state.sentence?.speaker||origin}</span><span>{sentenceIndex>=0?`${String(sentenceIndex+1).padStart(2,'0')} / ${String(episode.sentences.length).padStart(2,'0')}`:'LISTEN · LEARN · REPEAT'}</span></div>
      <div className="scene-english" data-caption-page={state.captionPage?.id||''}>{state.captionPage?state.captionPage.lines.map((line,i)=><div className="scene-caption-line" key={`${state.captionPage!.id}-${i}`}>{line.map((word,j)=><React.Fragment key={word.id}><span data-word-id={word.id} role={onWordSelect?"button":undefined} tabIndex={onWordSelect?0:undefined} title={onWordSelect&&!hideChinese?word.gloss_zh:undefined} onClick={onWordSelect?()=>onWordSelect(word):undefined} onKeyDown={onWordSelect?e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();onWordSelect(word);}}:undefined} className={`scene-word ${state.activeWord?.id===word.id?'scene-word-active':timeMs>=word.end_ms?'scene-word-read':''}`}>{word.text}</span>{j<line.length-1&&!/^[.,!?;:]/.test(line[j+1].text)?' ':''}</React.Fragment>)}</div>):<div className="scene-caption-wait">{outro?"Listen again. Notice something new.":"Listen. Learn. Understand."}</div>}</div>
      <div className="scene-progress-track"><span style={{width:`${progress*100}%`}}/></div>
    </div>
    <div className="scene-footer"><span>来源：{origin}{episode.episode.source_date?` · ${episode.episode.source_date}`:''}</span><span>英语 · 精听</span><span>{formatClock(timeMs/1000)} <i>/ {formatClock(episode.episode.duration_s)}</i></span></div>
  </div>;
}

export function ResponsiveScene(props:SceneProps){
  const ref=useRef<HTMLDivElement>(null),[width,setWidth]=useState(960);
  useEffect(()=>{const node=ref.current;if(!node)return;const observer=new ResizeObserver(entries=>setWidth(entries[0].contentRect.width));observer.observe(node);return()=>observer.disconnect();},[]);
  return <div ref={ref} className="scene-responsive" style={{aspectRatio:'16 / 9'}}><div className="scene-scaled" style={{transform:`scale(${width/1920})`}}><LessonScene {...props}/></div></div>;
}
