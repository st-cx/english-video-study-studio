import type {EpisodePackage,Sentence} from './shared';
import {formatClock} from './shared';

export function reviewClock(ms:number):string{return formatClock(ms/1000)+'.'+Math.floor(ms%1000/100);}

export type ReviewItem={id:string;kind:'文稿差异'|'词边界'|'时间异常';start_ms:number;end_ms:number;actual:string;reference:string;reason:string};
export function lessonReviewItems(episode:EpisodePackage):ReviewItem[]{
  const rows:ReviewItem[]=[],seen=new Set<string>(),duration=episode.episode.duration_s*1000;
  function add(kind:ReviewItem['kind'],r:{start_ms:number;end_ms:number;recognized?:string;reference?:string;text?:string;reason?:string}){
    if(!Number.isFinite(r.start_ms)||!Number.isFinite(r.end_ms)||r.start_ms<0||r.end_ms<r.start_ms||r.end_ms>duration)return;
    const actual=r.recognized||r.text||'',reference=r.reference||'',id=JSON.stringify([kind,r.start_ms,r.end_ms,actual,reference]);
    if(!seen.has(id)){seen.add(id);rows.push({id,kind,start_ms:r.start_ms,end_ms:r.end_ms,actual,reference,reason:r.reason||'请结合原声核对。'});}
  }
  for(const c of episode.chapters||[]){for(const r of c.reference_issues||[])add('文稿差异',r);for(const r of c.alignment_issues||[])add('词边界',r);}
  for(const r of episode.quality?.structural_issues||[])add('时间异常',r);
  return rows.sort((a,b)=>a.start_ms-b.start_ms||a.kind.localeCompare(b.kind));
}
export function adjacentSentence(sentences:Sentence[],timeMs:number,direction:-1|1):Sentence|null{
  if(direction===1)return sentences.find(s=>s.start_ms>timeMs+1)||null;
  const current=sentences.findIndex(s=>s.start_ms<=timeMs&&timeMs<s.end_ms);
  if(current>=0)return sentences[current-1]||null;
  return [...sentences].reverse().find(s=>s.end_ms<=timeMs)||null;
}
