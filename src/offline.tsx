import {createRoot} from 'react-dom/client';
import type {EpisodePackage} from './shared';
import {StudySession} from './StudySession';
import {LayoutGallery} from './LayoutGallery';
declare global {interface Window {EPISODE_DATA:EpisodePackage}}
createRoot(document.getElementById('study-root')!).render(new URLSearchParams(window.location.search).get('review')==='layout'?<LayoutGallery episode={window.EPISODE_DATA}/>:<StudySession episode={window.EPISODE_DATA}/>);
