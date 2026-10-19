import {build} from 'esbuild';
import {readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const built=await build({entryPoints:[path.join(root,'src/shared.ts')],bundle:true,platform:'node',format:'esm',write:false});
const {buildCaptionPages,textWidthEm,joinWords}=await import('data:text/javascript;base64,'+Buffer.from(built.outputFiles[0].text).toString('base64'));
const input=process.argv[2];
const pkg=JSON.parse(await readFile(input,'utf8'));
const pages=buildCaptionPages({...pkg,layout:undefined});
const seen=pages.flatMap(p=>p.words.map(w=>w.id));
if(seen.length!==pkg.words.length||new Set(seen).size!==pkg.words.length)throw Error('字幕页没有完整且唯一地覆盖词语。');
for(const p of pages){if(p.lines.length>2||p.lines.some(l=>textWidthEm(joinWords(l))>28.4))throw Error('字幕页超出固定字号排版范围：'+p.id);}
pkg.layout={version:1,pages:pages.map(p=>({id:p.id,sentence_id:p.sentence_id,start_ms:p.start_ms,end_ms:p.end_ms,lines:p.lines.map(l=>l.map(w=>w.id))}))};
await writeFile(input,JSON.stringify(pkg,null,2));
process.stdout.write(JSON.stringify({pages:pages.length,words:seen.length}));
