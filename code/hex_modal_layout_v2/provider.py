"""Task-scoped Vast lifecycle; credentials stay in a local mode-0600 file.

The watchdog destroys only the instance ID in this task's rental receipt.
No automatic rental retries or actions on other instances are permitted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request

BASE='https://console.vast.ai/api/v0/'
def api(key_file,method,path,data=None):
    key=key_file.read_text().strip()
    request=urllib.request.Request(BASE+path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)

def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def destroy(key_file,out,reason):
    rental=json.loads((out/'rental.json').read_text());identifier=rental['instance_id']
    response=api(key_file,'DELETE',f'instances/{identifier}/',{})
    receipt={'utc':now(),'instance_id':identifier,'reason':reason,
             'success':response.get('success',False),'message':response.get('msg')}
    write(out/'destruction.json',receipt)
    if not receipt['success']:raise RuntimeError('Provider did not confirm destruction')
    return receipt

def watchdog(key_file,out):
    rental=json.loads((out/'rental.json').read_text())
    deadline=rental['deadline_epoch']
    write(out/'watchdog.json',{'started_utc':now(),'instance_id':rental['instance_id'],'deadline_epoch':deadline})
    while time.time()<deadline:
        if (out/'destruction.json').exists():
            if json.loads((out/'destruction.json').read_text()).get('success'):return
        time.sleep(min(20,max(0,deadline-time.time())))
    # Retry temporary provider/network failures; never create replacement rentals.
    for attempt in range(10):
        try:destroy(key_file,out,'Absolute protocol rental deadline');return
        except Exception as exc:
            write(out/'watchdog-error.json',{'utc':now(),'attempt':attempt,'type':type(exc).__name__})
            time.sleep(20)
    raise RuntimeError('Rental destruction requires attention')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['watchdog','destroy'])
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='watchdog':watchdog(a.key_file,a.out)
    else:print(json.dumps(destroy(a.key_file,a.out,'Tasks finished; results collected')))
