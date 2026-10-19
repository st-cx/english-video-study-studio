import {bundle} from '@remotion/bundler';
import {openBrowser,renderStill,selectComposition} from '@remotion/renderer';
import {copyFile,mkdir,readFile,stat,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [input,assets]=process.argv.slice(2);
if(!input||!assets){console.error('用法：node scripts/check-layout.mjs <episode.json> <节目素材目录>');process.exit(2);}
const stamp=new Date().toISOString().replace(/[:.]/g,'-');
const runDir=path.join(root,'work','layout-audit',`${stamp}-${process.pid}`);
await mkdir(path.join(runDir,'public'),{recursive:true});
let browser;
let report;
const logs=[];
const browserLog=entry=>{
  const message=entry.text||'';
  logs.push(message);
  const marker=message.indexOf('ENGLISH_LAYOUT_AUDIT:');
  if(marker>=0){
    try{report=JSON.parse(message.slice(marker+'ENGLISH_LAYOUT_AUDIT:'.length));}catch{}
  }
};
try{
  const episode=JSON.parse(await readFile(path.resolve(input),'utf8'));
  const assetDir=path.resolve(assets);
  if(episode.episode.cover){
    const source=path.resolve(assetDir,episode.episode.cover);
    const relative=path.relative(assetDir,source);
    if(relative.startsWith('..')||path.isAbsolute(relative))throw Error('节目封面必须位于节目素材目录内。');
    const target=path.join(runDir,'public',relative);
    await mkdir(path.dirname(target),{recursive:true});
    await copyFile(source,target);
  }
  const browserCandidates=[process.env.REMOTION_BROWSER_EXECUTABLE,path.join(process.env.PROGRAMFILES||'C:\\Program Files','Google','Chrome','Application','chrome.exe'),path.join(process.env['PROGRAMFILES(X86)']||'C:\\Program Files (x86)','Microsoft','Edge','Application','msedge.exe')].filter(Boolean);
  let executable;
  for(const candidate of browserCandidates){try{if((await stat(candidate)).isFile()){executable=candidate;break;}}catch{}}
  const serveUrl=await bundle({entryPoint:path.join(root,'src','layout-audit-entry.tsx'),outDir:path.join(runDir,'bundle'),publicDir:path.join(runDir,'public'),onProgress:()=>{},enableCaching:true});
  browser=await openBrowser('chrome',{browserExecutable:executable,logLevel:'error'});
  const inputProps={episode};
  const composition=await selectComposition({serveUrl,id:'LayoutAudit',inputProps,puppeteerInstance:browser,timeoutInMilliseconds:120000,logLevel:'error',onBrowserLog:browserLog});
  await renderStill({serveUrl,composition,inputProps,puppeteerInstance:browser,output:path.join(runDir,'first-scene.png'),imageFormat:'png',timeoutInMilliseconds:120000,logLevel:'error',onBrowserLog:browserLog});
  if(!report||report.status!=='passed'||report.violations?.length)throw Error('画面检查未返回通过记录。');
  await writeFile(path.join(runDir,'report.json'),JSON.stringify({...report,episode_id:episode.episode.id,package_path:path.resolve(input)},null,2));
  console.log(JSON.stringify({status:'passed',cases:report.cases,checks:report.checks,report:path.join(runDir,'report.json')}));
}catch(error){
  await writeFile(path.join(runDir,'report.json'),JSON.stringify({...report,status:'failed',error:error instanceof Error?error.message:String(error)},null,2));
  console.error(`画面检查未通过：${error instanceof Error?error.message:String(error)}`);
  console.error(`检查记录：${path.join(runDir,'report.json')}`);
  process.exitCode=1;
}finally{
  await writeFile(path.join(runDir,'browser.log'),logs.join('\n'));
  if(browser)await browser.close({silent:true}).catch(()=>{});
}
