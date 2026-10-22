import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

// Isolated script test: no browser, account, network or real credentials.
const html=readFileSync(new URL('../public/bilibili-login.html',import.meta.url),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements=new Map();
const pending=[];
const context=vm.createContext({
  document:{getElementById(id){if(!elements.has(id))elements.set(id,{textContent:'',hidden:true,disabled:false,src:''});return elements.get(id);},documentElement:{classList:{add(){}}}},
  location:{search:'',reload(){}},URLSearchParams,Date,JSON,Error,Math,
  setInterval(){return 1;},clearInterval(){},setTimeout(){return 2;},clearTimeout(){},
  window:{addEventListener(){}},
  fetch(path,options){return new Promise(resolve=>pending.push({path,options,reply(value){resolve({ok:true,json:async()=>value});}}));},
});
vm.runInContext(script,context);
async function drain(){for(let i=0;i<15;i++)await Promise.resolve();}
await drain();
assert.equal(pending[0].path,'/api/publishing');
pending.shift().reply({account:{configured:false}});await drain();
pending.shift().reply({login_id:'old',status:'scan',expires_in:180,qr_image:'data:image/png;base64,old'});await drain();
assert.equal(elements.get('qr').src,'data:image/png;base64,old');

const oldPoll=vm.runInContext('poll()',context);await drain();
assert.equal(pending[0].path,'/api/publishing/login/poll');
const oldRequest=pending.shift();
const renewal=vm.runInContext('generate(true)',context);await drain();
pending.shift().reply({login_id:'new',status:'scan',expires_in:180,qr_image:'data:image/png;base64,new'});await renewal;
assert.equal(elements.get('qr').src,'data:image/png;base64,new');
oldRequest.reply({status:'expired',configured:false});await oldPoll;
assert.equal(vm.runInContext('session?.login_id',context),'new','a stale poll cannot clear a newly generated session');
assert.equal(elements.get('qr').hidden,false,'a stale expiry cannot hide the new code');

const confirmPoll=vm.runInContext('poll()',context);await drain();
pending.shift().reply({status:'confirm',configured:true});await drain();
pending.shift().reply({account:{configured:true,login_diagnostic:{status:'confirm'}}});await confirmPoll;
assert.match(elements.get('status').textContent,/手机/,'the current session can still receive confirmation');
assert.equal(elements.get('qr').hidden,false,'an existing credential file cannot hide a new login code');

const disconnectedPoll=vm.runInContext('poll()',context);await drain();
pending.shift().reply({status:'expired',configured:true});await drain();
assert.equal(pending[0]?.path,'/api/publishing');
pending.shift().reply({account:{configured:true,login_diagnostic:{status:'done'}}});await drain();
assert.equal(pending[0]?.path,'/api/publishing/verify','a connected account is verified before rendering expiry');
pending.shift().reply({verified:true});await disconnectedPoll;
assert.equal(elements.get('success').hidden,false);
assert.equal(elements.get('qr').hidden,true);
console.log(JSON.stringify({status:'passed',checks:11,isolated:true}));
