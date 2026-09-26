"""Post-run export of every timing row and primary fixed-work distributions."""
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
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    result=json.loads((a.rental/'remote/run-01/result.json').read_text())
    replay=json.loads((a.rental/'local-verification.json').read_text())
    p=json.loads((a.rental/'remote/source/protocol.json').read_text())
    assert replay['evidence_valid']
    fields=['case','path','round','position','calls','gpu_ms_per_call','host_ms_per_call']
    with (a.out/'all-timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in result['timings'])
    comparisons=replay['decision']['comparisons']
    if comparisons:
        with (a.out/'all-comparisons.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    if replay['decision']['complete']:
        paths=p['paths']
        names=['Three stage','Fused indexed','Fused analytic','Node gather','Dense 8','Modal 8','Symmetric 8','Dense 32','Modal 32']
        fig,axes=plt.subplots(1,2,figsize=(11,5.8),layout='constrained')
        for ax,case in zip(axes,[c['id'] for c in p['cases'] if c['primary']]):
            medians=[];low=[];high=[]
            for path in paths:
                values=[r['gpu_ms_per_call'] for r in result['timings'] if r['case']==case and r['path']==path]
                middle=statistics.median(values);medians.append(middle)
                low.append(middle-min(values));high.append(max(values)-middle)
            colors=['#a0a7ae' if path not in p['candidates'] else '#236b99' if path=='modal8' else '#b36626' for path in paths]
            ax.barh(np.arange(len(paths)),medians,xerr=[low,high],color=colors,error_kw={'capsize':3,'linewidth':1})
            ax.set_yticks(np.arange(len(paths)),names);ax.invert_yaxis()
            ax.set_title(case);ax.set_xlabel('Complete product time (ms; lower is better)')
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(0,max(medians)*1.20)
            for i,v in enumerate(medians):
                ax.text(v+max(medians)*.025,i,f'{v:.3f}',va='center',fontsize=9)
        fig.suptitle('Fixed-input operator development on one RTX 4090',fontsize=15)
        fig.supxlabel('Bars: medians of 11 randomized timing blocks; whiskers: observed ranges.\n'
            'Each block has 20 identical-input products. Preparation and transfers excluded; no application-time claim.',fontsize=9)
        for suffix in ('png','svg'):fig.savefig(a.out/('primary-products.'+suffix),dpi=180)
        plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.out/'provenance.json').write_text(json.dumps({'result_sha256':sha(a.rental/'remote/run-01/result.json'),
        'replay_sha256':sha(a.rental/'local-verification.json'),'rows_exported':len(result['timings']),
        'matplotlib':matplotlib.__version__,'numpy':np.__version__,'whiskers':'observed ranges, not confidence intervals',
        'scope':'Fixed-work product timings, no end-to-end or novelty claim'},indent=2)+'\n')
    print(json.dumps({'rows_exported':len(result['timings']),'complete':replay['decision']['complete']}))


if __name__=='__main__':main()
