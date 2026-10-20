import {captionAdvances} from './fontMetrics';
export type TimedWord = {id:string;sentence_id:string;index:number;text:string;start_ms:number;end_ms:number;gloss_zh:string;probability:number;focus?:boolean};
export type Sentence = {id:string;start_ms:number;end_ms:number;text:string;translation:string;speaker:string;word_ids:string[];card_ids:string[]};
export type TimedPhrase = {id:string;text:string;gloss_zh:string;word_ids:string[];sentence_id:string;study?:boolean};
export type StudyCard = {id:string;type:'vocab'|'grammar';title:string;meaning:string;structure:string;example:string;translation:string;word_ids:string[];sentence_ids:string[];start_ms:number;end_ms:number;display?:boolean;show_meaning?:boolean;points?:{label:string;text:string}[]};
export type EpisodePackage = {
  version:number;
  episode:{id:string;title:string;title_zh:string;source_url:string;audio_url:string;duration_s:number;source_date:string;cover:string;audio_file:string;[key:string]:unknown};
  sentences:Sentence[];words:TimedWord[];phrases?:TimedPhrase[];cards:StudyCard[];
  layout?:{version:number;pages:{id:string;sentence_id:string;start_ms:number;end_ms:number;lines:string[][];translation?:string}[]};
  source?:{kind:string;name?:string;url?:string;transcript_kind?:string};
  media?:{video_file?:string;audio_file:string;duration_s:number;source_offset_ms?:number;audio_stream_index?:number};
  chapters?:{id:string;title:string;start_ms:number;end_ms:number;status:'ready'|'review'|'pending';teaching_status?:string}[];
  artifacts?:{study:'pending'|'ready';video:'pending'|'ready'|'not_requested'|'failed'};
  quality:{needs_review:boolean;flags:string[];source_similarity:number|null};
  review:{recognized_transcript:string;official_transcript:string;approved:boolean;[key:string]:unknown};render:Record<string,unknown>;
};
export type CaptionPage = {id:string;sentence_id:string;words:TimedWord[];lines:TimedWord[][];start_ms:number;end_ms:number;translation?:string};
export type GlossUnit = {id:string;text:string;gloss_zh:string;word_ids:string[];phrase:boolean};
export type LessonState = {sentence:Sentence|null;sentenceWords:TimedWord[];visibleWords:TimedWord[];captionPage:CaptionPage|null;activeWord:TimedWord|null;lastSpokenWord:TimedWord|null;glossUnit:GlossUnit|null;card:StudyCard|null};

export function cardHeading(card:StudyCard,hideChinese=false):string {
  if(!hideChinese)return card.title;
  const english=card.title.replace(/[\u2e80-\u9fff\uff00-\uffef]/g,'').replace(/\s+/g,' ').replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9.?!]+$/g,'').trim();
  return /[A-Za-z]/.test(english)?english:'Sentence focus';
}

export function joinWords(words:Pick<TimedWord,'text'>[]):string {
  return words.map((w,i)=>(i&&!/^[.,!?;:]/.test(w.text)?' ':'')+w.text).join('');
}
export function textWidthEm(text:string):number {
  return [...text].reduce((sum,c)=>sum+(captionAdvances[c]??(/[\u2e80-\u9fff]/.test(c)?1:.6))+.1/48,0);
}
const LINE_WIDTH = 28.0;
function balancedLines(units:TimedWord[][]):TimedWord[][]|null {
  const flat=units.flat();
  if(textWidthEm(joinWords(flat))<=LINE_WIDTH)return [flat];
  let best:TimedWord[][]|null=null,bestCost=Infinity;
  for(let i=1;i<units.length;i++){
    const a=units.slice(0,i).flat(),b=units.slice(i).flat();
    const aw=textWidthEm(joinWords(a)),bw=textWidthEm(joinWords(b));
    if(aw>LINE_WIDTH||bw>LINE_WIDTH)continue;
    const orphanPenalty=b.length===1?20:0;
    const tail=a.at(-1)!.text;
    const clauseBonus=/[,;:]$/.test(tail)?6:0;
    const dangling=/^(i|a|an|the|to|of|and|or|our|this|with|don't|can|would|because)$/i.test(tail)?12:0;
    const cost=Math.abs(aw-bw)+orphanPenalty+dangling-clauseBonus;
    if(cost<bestCost){best=[a,b];bestCost=cost;}
  }
  return best;
}
// Prepare complete pages once: highlighting never moves a page's word boundaries.
export function buildCaptionPages(pkg:EpisodePackage):CaptionPage[]{
  if(pkg.layout?.version===1){
    const byId=new Map(pkg.words.map(w=>[w.id,w]));
    return pkg.layout.pages.map(p=>{const lines=p.lines.map(line=>line.map(id=>{const word=byId.get(id);if(!word)throw Error('字幕页引用了不存在的词语：'+id);return word;}));return {...p,lines,words:lines.flat()};});
  }
  const pages:CaptionPage[]=[];
  for(const sentence of pkg.sentences){
    const words=pkg.words.filter(w=>w.sentence_id===sentence.id).sort((a,b)=>a.index-b.index);
    const phrases=(pkg.phrases||[]).filter(p=>p.sentence_id===sentence.id).sort((a,b)=>b.word_ids.length-a.word_ids.length);
    const units:TimedWord[][]=[];
    for(let i=0;i<words.length;){
      const phrase=phrases.find(p=>p.word_ids[0]===words[i].id&&p.word_ids.every((id,k)=>words[i+k]?.id===id));
      const size=phrase?.word_ids.length||1;
      const group=words.slice(i,i+size);
      if(/^[.,!?;:]/.test(group[0].text)&&units.length)units[units.length-1].push(...group);
      else units.push(group);
      i+=size;
    }
    const costs=Array(units.length+1).fill(Infinity),choices=new Map<number,{end:number;lines:TimedWord[][]}>();
    costs[units.length]=0;
    for(let start=units.length-1;start>=0;start--){
      for(let end=start+1;end<=units.length;end++){
        const chunk=units.slice(start,end),flat=chunk.flat(),lines=balancedLines(chunk);
        if(!lines)break;
        const width=textWidthEm(joinWords(flat)),tail=flat.at(-1)!.text;
        const orphan=flat.length<4&&words.length>=8?75:0;
        const dangling=/^(a|an|the|to|of|and|or|our|this|with)$/i.test(tail)?35:0;
        const boundary=/[,;:]$/.test(tail)?6:0;
        const nextClause=units[end]&&/^(along|but|because|on|and|as)$/i.test(units[end][0].text)?4:0;
        const score=120+Math.pow(LINE_WIDTH*2-width,2)*.022+orphan+dangling-boundary-nextClause+costs[end];
        if(score<costs[start]){costs[start]=score;choices.set(start,{end,lines});}
      }
    }
    for(let start=0;start<units.length;){
      const choice=choices.get(start)||{end:start+1,lines:[units[start]]};
      const {end,lines}=choice;
      const flat=units.slice(start,end).flat();
      pages.push({id:sentence.id+'-p'+start,sentence_id:sentence.id,words:flat,lines,start_ms:flat[0].start_ms,end_ms:flat.at(-1)!.end_ms});
      start=end;
    }
  }
  return pages;
}
type Prepared={sentences:Sentence[];pages:CaptionPage[];cards:StudyCard[];cardSegments:{start_ms:number;end_ms:number;card:StudyCard}[];wordsBySentence:Map<string,TimedWord[]>;pagesBySentence:Map<string,CaptionPage[]>;phrasesByWord:Map<string,TimedPhrase[]>;sentenceById:Map<string,Sentence>;wordById:Map<string,TimedWord>;sentenceIndex:Map<string,number>;cardIndex:Map<string,number>};
const prepared=new WeakMap<EpisodePackage,Prepared>();
function prepare(pkg:EpisodePackage):Prepared{
  let result=prepared.get(pkg);if(result)return result;
  const wordsBySentence=new Map<string,TimedWord[]>();
  for(const w of pkg.words){const group=wordsBySentence.get(w.sentence_id)||[];group.push(w);wordsBySentence.set(w.sentence_id,group);}
  for(const words of wordsBySentence.values())words.sort((a,b)=>a.index-b.index);
  const pages=buildCaptionPages(pkg),pagesBySentence=new Map<string,CaptionPage[]>(),phrasesByWord=new Map<string,TimedPhrase[]>();
  for(const p of pages){const a=pagesBySentence.get(p.sentence_id)||[];a.push(p);pagesBySentence.set(p.sentence_id,a);}
  for(const phrase of pkg.phrases||[])if(textWidthEm(phrase.text)<=16&&phrase.gloss_zh.length<=22)for(const id of phrase.word_ids){const a=phrasesByWord.get(id)||[];a.push(phrase);phrasesByWord.set(id,a);}
  for(const a of phrasesByWord.values())a.sort((a,b)=>b.word_ids.length-a.word_ids.length);
  const cards=pkg.cards.filter(c=>c.display!==false).sort((a,b)=>a.start_ms-b.start_ms);
  const boundaries=[...new Set(cards.flatMap(c=>[c.start_ms,c.end_ms]))].sort((a,b)=>a-b),stack:StudyCard[]=[],cardSegments:Prepared['cardSegments']=[];let ci=0;
  for(let bi=0;bi<boundaries.length-1;bi++){const t=boundaries[bi];while(ci<cards.length&&cards[ci].start_ms<=t)stack.push(cards[ci++]);while(stack.length&&stack[stack.length-1].end_ms<=t)stack.pop();if(stack.length)cardSegments.push({start_ms:t,end_ms:boundaries[bi+1],card:stack[stack.length-1]});}
  result={sentences:[...pkg.sentences].sort((a,b)=>a.start_ms-b.start_ms),pages,cards,cardSegments,wordsBySentence,pagesBySentence,phrasesByWord,sentenceById:new Map(pkg.sentences.map(s=>[s.id,s])),wordById:new Map(pkg.words.map(w=>[w.id,w])),sentenceIndex:new Map(pkg.sentences.map((s,i)=>[s.id,i])),cardIndex:new Map(cards.map((c,i)=>[c.id,i]))};
  prepared.set(pkg,result);return result;
}
export function deriveLessonState(pkg:EpisodePackage,timeMs:number):LessonState{
  const data=prepare(pkg);
  const outro=timeMs>(data.sentences.at(-1)?.end_ms??Infinity)+1200;
  const sentence=outro?null:data.sentences[timeIndex(data.sentences,timeMs)]||null;
  const sentenceWords=sentence?data.wordsBySentence.get(sentence.id)||[]:[];
  const word=sentenceWords[timeIndex(sentenceWords,timeMs)]||null;
  const activeWord=word&&timeMs<word.end_ms?word:null;
  const lastSpokenWord=word;
  const candidates=data.pagesBySentence.get(sentence?.id||'')||[];
  const captionPage=candidates[timeIndex(candidates,timeMs)]||candidates[0]||null;
  const phrase=lastSpokenWord?data.phrasesByWord.get(lastSpokenWord.id)?.[0]:null;
  const glossUnit:GlossUnit|null=phrase?{...phrase,phrase:true}:lastSpokenWord?{id:lastSpokenWord.id,text:lastSpokenWord.text.replace(/^[.,!?;:]+|[.,!?;:]+$/g,''),gloss_zh:lastSpokenWord.gloss_zh,word_ids:[lastSpokenWord.id],phrase:false}:null;
  const segment=data.cardSegments[timeIndex(data.cardSegments,timeMs)];
  const card=outro?null:segment&&timeMs<segment.end_ms?segment.card:null;
  return {sentence,sentenceWords,visibleWords:captionPage?.words||[],captionPage,activeWord,lastSpokenWord,glossUnit,card};
}
export function timeIndex<T extends {start_ms:number}>(items:T[],ms:number):number{let low=0,high=items.length;while(low<high){const mid=(low+high)>>>1;if(items[mid].start_ms<=ms)low=mid+1;else high=mid;}return low-1;}
export function episodeIndex(pkg:EpisodePackage){return prepare(pkg);}
export function sourceName(pkg:EpisodePackage):string{if(pkg.source?.name)return pkg.source.name;if(pkg.source?.kind==='local')return '本地素材';try{return new URL(pkg.source?.url||pkg.episode.source_url).hostname.replace(/^www\./,'');}catch{return '学习素材';}}
export function mediaFiles(pkg:EpisodePackage){return {video:pkg.media?.video_file||String(pkg.episode.video_file||''),audio:pkg.media?.audio_file||pkg.episode.audio_file||'audio.mp3'};}
export function episodeChapters(pkg:EpisodePackage){return pkg.chapters?.length?pkg.chapters:[{id:'all',title:'完整内容',start_ms:0,end_ms:pkg.episode.duration_s*1000,status:'ready' as const}];}
export function formatClock(seconds:number):string{
  const n=Math.max(0,Math.floor(seconds));return `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toString().padStart(2,'0')}`;
}
