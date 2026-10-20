import type {EpisodePackage} from './shared';

export function episodeStudySections(episode:EpisodePackage){
  const chapters=episode.chapters?.length?episode.chapters:[{id:'all',title:'完整内容',start_ms:0,end_ms:episode.episode.duration_s*1000,status:'ready' as const}];
  if(episode.study_sections?.status!=='ready'||!episode.study_sections.items.length)return chapters.map(c=>({...c,summary:''}));
  return episode.study_sections.items.map(section=>{
    const affected=chapters.filter(c=>c.start_ms<section.end_ms&&c.end_ms>section.start_ms);
    const status=affected.some(c=>c.status==='review')?'review' as const:affected.some(c=>c.status==='pending')?'pending' as const:'ready' as const;
    return {...section,status};
  });
}

export function followOffset(current:number,viewport:number,start:number,size:number,total:number):number{
  const margin=14,max=Math.max(0,total-viewport);
  if(start>=current+margin&&start+size<=current+viewport-margin)return current;
  const target=size>=viewport-2*margin?start-margin:start-(viewport-size)*.32;
  return Math.max(0,Math.min(max,target));
}
