import {useCallback,useEffect,useRef,useState} from 'react';
import type {RefObject} from 'react';
import {followOffset} from './studyNavigation';

/** Scroll a list only. User browsing never moves the document or playback. */
export function usePlaybackFollow(container:RefObject<HTMLDivElement|null>,target:string|null,scope:string,axis:'vertical'|'horizontal'='vertical'){
  const [enabled,setEnabled]=useState(true),[suspended,setSuspended]=useState(false),[revision,setRevision]=useState(0);
  const previousScope=useRef(scope);
  useEffect(()=>{if(previousScope.current!==scope){previousScope.current=scope;setSuspended(false);}},[scope]);
  const resume=useCallback(()=>{setEnabled(true);setSuspended(false);setRevision(v=>v+1);},[]);
  const toggle=useCallback((value:boolean)=>{setEnabled(value);setSuspended(false);},[]);
  useEffect(()=>{
    const node=container.current;if(!node)return;
    const browse=()=>setSuspended(true);
    const keyboard=(event:KeyboardEvent)=>{if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','PageUp','PageDown','Home','End',' '].includes(event.key))browse();};
    const pointer=(event:PointerEvent)=>{const box=node.getBoundingClientRect();if(axis==='vertical'?event.clientX>=box.left+node.clientWidth:event.clientY>=box.top+node.clientHeight)browse();};
    node.addEventListener('wheel',browse,{passive:true});node.addEventListener('touchmove',browse,{passive:true});node.addEventListener('keydown',keyboard);
    node.addEventListener('pointerdown',pointer);
    return()=>{node.removeEventListener('wheel',browse);node.removeEventListener('touchmove',browse);node.removeEventListener('keydown',keyboard);node.removeEventListener('pointerdown',pointer);};
  },[container,axis]);
  useEffect(()=>{
    const list=container.current;if(!list||!target||!enabled||suspended)return;
    const item=list.querySelector<HTMLElement>(`[data-follow-id="${CSS.escape(target)}"]`);if(!item)return;
    const scroll=()=>{
      const box=list.getBoundingClientRect(),row=item.getBoundingClientRect(),horizontal=axis==='horizontal';
      const current=horizontal?list.scrollLeft:list.scrollTop,viewport=horizontal?list.clientWidth:list.clientHeight;
      const start=(horizontal?row.left-box.left:row.top-box.top)+current;
      const next=followOffset(current,viewport,start,horizontal?row.width:row.height,horizontal?list.scrollWidth:list.scrollHeight);
      if(Math.abs(next-current)<1)return;
      const behavior=window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant' as const:'smooth' as const;
      list.scrollTo(horizontal?{left:next,behavior}:{top:next,behavior});
    };
    const frame=requestAnimationFrame(scroll),observer=new ResizeObserver(scroll);observer.observe(list);observer.observe(item);
    return()=>{cancelAnimationFrame(frame);observer.disconnect();};
  },[container,target,scope,enabled,suspended,axis,revision]);
  return {enabled,suspended,toggle,resume};
}
