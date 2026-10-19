import {build} from 'esbuild';
await build({entryPoints:['src/offline.tsx'],bundle:true,minify:true,format:'iife',target:'es2022',outfile:'dist/offline/lesson.js',define:{'process.env.NODE_ENV':'"production"'},jsx:'automatic',legalComments:'none'});
