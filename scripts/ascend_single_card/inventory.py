"""Read-only inventory and preservation manifest for the real NPU acceptance."""
import hashlib, importlib, importlib.metadata as md, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path('/root/lmetric-repro')
OUT = ROOT/'results/ascend_single_card_validation'
OUT.mkdir(exist_ok=True)
def run(cmd):
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 return dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():
 result={}
 for relative in ['engines/vllm','blitz-router','/vllm-workspace/vllm','/vllm-workspace/vllm-ascend']:
  repo=ROOT/relative
  row={key:run(['git','-C',str(repo),*args]) for key,args in {'status':['status','--short','--branch'],'branches':['branch','-avv'],'head':['rev-parse','HEAD'],'diff':['diff','--binary'],'files':['ls-files','-m','-o','--exclude-standard']}.items()}
  row['hashes']={f:sha(repo/f) for f in row['files']['stdout'].splitlines() if (repo/f).is_file()}
  result[relative]=row
 return result
if __name__=='__main__':
 name=sys.argv[1] if len(sys.argv)>1 else 'initial'
 if name=='initial':
  data={'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'python':sys.version,'executable':sys.executable,'root_git':run(['git','status','--short','--branch']),'npu_smi':run(['npu-smi','info']),'repositories':manifest(),'packages':{}}
  for package,module in [('torch','torch'),('torch-npu','torch_npu'),('vllm','vllm'),('vllm-ascend','vllm_ascend')]:
   d=md.distribution(package);m=importlib.import_module(module)
   data['packages'][package]={'version':d.version,'module_version':getattr(m,'__version__',None),'path':m.__file__,'metadata':str(d._path),'direct_url':d.read_text('direct_url.json')}
  import torch,torch_npu
  data['npu']={'available':torch.npu.is_available(),'device_count':torch.npu.device_count(),'name':torch.npu.get_device_name(0)}
  candidates=[]
  for path in sorted(Path('/models').glob('*/config.json')):
   model=path.parent;weights=list(model.glob('*.safetensors'))+list(model.glob('pytorch_model*.bin'))
   candidates.append({'path':str(model),'realpath':str(model.resolve()),'config_sha256':sha(path),'weight_files':len(weights),'weight_bytes':sum(p.stat().st_size for p in weights)})
  data['local_models']=candidates
  data['selected_model']={str(p):{'bytes':p.stat().st_size,'sha256':sha(p) if p.suffix=='.json' else None} for p in Path('/models/Qwen3-0.6B').iterdir() if p.is_file()}
 else:
  data={'repositories':manifest(),'npu_smi':run(['npu-smi','info'])}
 (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 print('Saved',OUT/(name+'.json'))
