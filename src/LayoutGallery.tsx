import type {EpisodePackage} from './shared';
import {buildCaptionPages} from './shared';
import {ResponsiveScene} from './LessonScene';

export function LayoutGallery({episode}:{episode:EpisodePackage}){
  const times=[...new Set([0,...buildCaptionPages(episode).map(p=>p.start_ms+70),...episode.cards.filter(c=>c.display!==false).map(c=>c.start_ms+600),...episode.words.filter(w=>w.gloss_zh.length>20).slice(0,12).map(w=>w.start_ms+10),episode.episode.duration_s*1000-500])].sort((a,b)=>a-b);
  return <main className="layout-gallery" style={{maxWidth:1000,margin:'auto',padding:16}}><h1>全页画面检查</h1><p>{times.length} 个字幕与讲解边界；仅用于制作检查。</p>{times.map(t=><section className="layout-case" data-check-ms={t} key={t} style={{marginBottom:24}}><p>{(t/1000).toFixed(2)}s</p><ResponsiveScene episode={episode} timeMs={t} coverSrc={'./'+episode.episode.cover}/></section>)}</main>;
}
