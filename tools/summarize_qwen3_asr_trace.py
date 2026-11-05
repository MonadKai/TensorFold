"""Summarize asr-* NVTX workload ranges from a Nsight Systems 2025.3.2 SQLite export.

Usage: python3 tools/summarize_qwen3_asr_trace.py TRACE.sqlite SUMMARY.json
Raw trace databases can contain private paths/environment; publish only the summary.
CUDA activity unions are elapsed device activity, not hardware occupancy measurements.
"""
import hashlib,json,sqlite3,sys
from pathlib import Path
path=Path(sys.argv[1]);out=Path(sys.argv[2]);c=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)
devices=[row[0] for row in c.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL")]
assert len(devices)==1,"this summary requires one CUDA device"
strings=dict(c.execute('select id,value from StringIds'))
ranges=[(s,e,text or strings.get(tid,'')) for s,e,text,tid in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null') if (text or strings.get(tid,'')).startswith('asr-')]
assert ranges,'no named workload ranges'
def union(xs):
 total=0;end=-1
 for a,b in sorted(xs):
  if b>end:total+=b-max(a,end);end=b
 return total/1e9
rows=[]
for begin,end,label in ranges:
 activity=[];kernel=[];copies=[];sets=[]
 for table,target in [('CUPTI_ACTIVITY_KIND_KERNEL',kernel),('CUPTI_ACTIVITY_KIND_MEMCPY',copies),('CUPTI_ACTIVITY_KIND_MEMSET',sets)]:
  if not c.execute('select count(*) from sqlite_master where name=?',(table,)).fetchone()[0]:continue
  for a,b in c.execute(f'select start,end from {table} where end>? and start<?',(begin,end)):
   pair=(max(a,begin),min(b,end));activity.append(pair);target.append(pair)
 top=[]
 for name,count,ns in c.execute('select demangledName,count(*),sum(min(end,?)-max(start,?)) from CUPTI_ACTIVITY_KIND_KERNEL where end>? and start<? group by demangledName order by 3 desc limit 12',(end,begin,begin,end)):
  top.append({'kernel':strings[name],'launches':count,'summed_seconds':ns/1e9})
 rows.append({'range':label,'range_wall_seconds':(end-begin)/1e9,'cuda_activity_union_seconds':union(activity),'kernel_union_seconds':union(kernel),'copy_union_seconds':union(copies),'memset_union_seconds':union(sets),'kernel_launches':len(kernel),'copy_operations':len(copies),'top_kernels':top})
# Kind labels and operation values are read from the actual exported schema, not guessed.
alloc=dict((name,kind) for kind,name,label in c.execute('select * from ENUM_CUDA_DEV_MEM_EVENT_OPER'))['CUDA_DEV_MEM_EVENT_OPR_ALLOCATION']
memkind=dict((kind,label) for kind,name,label in c.execute('select * from ENUM_CUDA_MEM_KIND'))
active={};totals={};peaks={};duplicates=0;unknown_frees=0
for stamp,pid,device,context,address,size,kind,op in c.execute('select start,globalPid,deviceId,contextId,address,bytes,memKind,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start,rowid'):
 key=(pid,device,context,kind,address);group=memkind[kind]
 if op==alloc:
  if key in active:duplicates+=1;totals[group]-=active[key]
  active[key]=size;totals[group]=totals.get(group,0)+size;peaks[group]=max(peaks.get(group,0),totals[group])
 else:
  if key not in active:unknown_frees+=1
  totals[group]=totals.get(group,0)-active.pop(key,0)
result={'schema':'asr-cuda-profile-v1','tool':'Nsight Systems2025.3.2','activity_devices':devices,'trace_sqlite_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'workloads':rows,'captured_allocation_peak_bytes_by_kind':peaks,'duplicate_allocation_events':duplicates,'unknown_free_events':unknown_frees,'limitations':['Profiling overhead and workload-range client preparation are included; use separate unprofiled cells for speed claims.','Activity is the union of captured kernels, copies and memsets, not SM occupancy or total-system GPU utilization.','Activity subcategories can overlap; do not add their union durations.','Allocation peaks cover the full traced process lifetime including startup and warmups; allocation bytes are not physical-memory residency.']}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'workloads':[{k:v for k,v in r.items() if k!='top_kernels'} for r in rows],'allocation_peaks':peaks}))
