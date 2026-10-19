import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {performance} from 'node:perf_hooks';
const built=await build({entryPoints:['src/shared.ts'],bundle:true,platform:'node',format:'esm',write:false});
const {deriveLessonState,episodeIndex,mediaFiles,sourceName,timeIndex}=await import('data:text/javascript;base64,'+Buffer.from(built.outputFiles[0].text).toString('base64'));
const word=(id,sid,text,start,end)=>({id,sentence_id:sid,index:0,text,start_ms:start,end_ms:end,gloss_zh:'语境含义',probability:1});
const pkg={version:3,episode:{id:'test',title:'Test',title_zh:'测试',source_url:'',duration_s:4,audio_file:'old.mp3'},sentences:[{id:'s0',start_ms:0,end_ms:1800,text:'Hello world.',translation:'你好，世界。',word_ids:['w0','w1']},{id:'s1',start_ms:2000,end_ms:3800,text:'Listen again.',translation:'再听一次。',word_ids:['w2','w3']}],words:[word('w0','s0','Hello',0,700),{...word('w1','s0','world.',900,1600),index:1},word('w2','s1','Listen',2000,2700),{...word('w3','s1','again.',2900,3600),index:1}],cards:[{id:'outer',type:'vocab',title:'world',meaning:'世界',start_ms:0,end_ms:3900},{id:'inner',type:'grammar',title:'Short interruption',meaning:'短暂插入',start_ms:2100,end_ms:2500}],phrases:[{id:'p1',text:'Listen again',gloss_zh:'再听一次',word_ids:['w2','w3'],sentence_id:'s1'}]};
assert.equal(timeIndex(pkg.sentences,-1),-1);
assert.equal(deriveLessonState(pkg,0).activeWord.id,'w0');
assert.equal(deriveLessonState(pkg,800).activeWord,null);
assert.equal(deriveLessonState(pkg,800).lastSpokenWord.id,'w0');
assert.equal(deriveLessonState(pkg,2200).sentence.id,'s1');
assert.equal(deriveLessonState(pkg,2200).glossUnit.id,'p1');
assert.equal(deriveLessonState(pkg,2200).card.id,'inner');
assert.equal(deriveLessonState(pkg,2600).card.id,'outer');
assert.equal(deriveLessonState(pkg,200).sentence.id,'s0');
assert.equal(mediaFiles(pkg).audio,'old.mp3');
const v4={...pkg,version:4,source:{kind:'local',name:'个人课程'},media:{video_file:'source.mp4',audio_file:'audio.mp3',duration_s:4}};
assert.equal(mediaFiles(v4).video,'source.mp4');
assert.equal(mediaFiles(v4).audio,'audio.mp3');
assert.equal(sourceName(v4),'个人课程');
assert.equal(episodeIndex(v4),episodeIndex(v4));
const long={...pkg,episode:{...pkg.episode,duration_s:20000},phrases:[],cards:[],sentences:[],words:[]};
for(let i=0;i<5000;i++){long.sentences.push({...pkg.sentences[0],id:'s'+i,start_ms:i*4000,end_ms:i*4000+3500,word_ids:['a'+i]});long.words.push(word('a'+i,'s'+i,'Context.',i*4000,i*4000+3000));}
episodeIndex(long);const started=performance.now();
for(let i=0;i<10000;i++){const index=(i*37)%5000;assert.equal(deriveLessonState(long,index*4000+100).sentence.id,'s'+index);}
console.log(JSON.stringify({status:'passed',assertions:10014,long_sentences:5000,random_seeks:10000,lookup_ms:Math.round(performance.now()-started)}));
