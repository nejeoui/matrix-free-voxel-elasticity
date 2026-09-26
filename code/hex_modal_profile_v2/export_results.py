"""Export complete diagnostic measurements; ranges are not confidence intervals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rental',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    replay=json.loads((args.rental/'local-verification.json').read_text())
    result=json.loads((args.rental/'remote/run-01/result.json').read_text())
    assert replay['evidence_valid'] and replay['states_replayed']==90
    assert result['status']=='completed' and len(result['solves'])==70
    assert all(x['physical_passed'] for x in result['solves'])
    paths=['fused_ai_fp64','fused_fp64','node_fp64','dense8','modal8']
    labels=['Fused analytic','Fused indexed','Node gather','Dense 8-thread','Modal 8-thread']
    columns=['case','path','profiled','replicate','iterations','matvec_calls','solver_seconds',
             'product_event_seconds','independent_true_residual','physical_passed']
    with (args.out/'all-solves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in result['solves'])
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained')
    for ax,case in zip(axes,['optimized-q64','optimized-q96']):
        medians=[];low=[];high=[]
        for path in paths:
            values=[r['solver_seconds'] for r in result['solves']
                    if r['case']==case and r['path']==path and not r['profiled']]
            assert len(values)==3
            middle=statistics.median(values);medians.append(middle)
            low.append(middle-min(values));high.append(max(values)-middle)
        y=np.arange(len(paths))
        ax.barh(y,medians,xerr=[low,high],color=['#8b98a4']*4+['#276c9b'],
                error_kw={'capsize':3,'linewidth':1})
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(case)
        ax.set_xlabel('Warm solve time (seconds; lower is better)')
        ax.set_xlim(0,max(medians)*1.20)
        for i,value in enumerate(medians):ax.text(value+max(medians)*.025,i,f'{value:.3f}',va='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Prepared-state timing diagnostic',fontsize=15)
    fig.supxlabel('Bars: medians of 3 solves; whiskers: observed ranges, not confidence intervals.\n'
                  'Startup and preparation excluded. The combined profiling gate failed.',fontsize=9)
    for suffix in ['png','svg']:fig.savefig(args.out/('primary-timings.'+suffix),dpi=180)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out/'provenance.json').write_text(json.dumps({'result_sha256':sha(args.rental/'remote/run-01/result.json'),
        'replay_sha256':sha(args.rental/'local-verification.json'),'matplotlib':matplotlib.__version__,
        'numpy':np.__version__,'rows_exported':70,'bootstrap_or_confidence_interval':False,
        'publication_claim':False},indent=2)+'\n')
    print(json.dumps({'rows_exported':70,'figures':['primary-timings.png','primary-timings.svg']}))


if __name__=='__main__':main()
