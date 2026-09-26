"""Audit the completed iteration and seal evidence without changing older seals."""
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/rescope/hex-modal-layout-v2-20260923'
REPORT=ROOT/'results/reports/scientific-report-hex-modal-layout-v2-20260923'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(p.read_text())


def write(p,data):
    assert not p.exists(),p
    p.write_text(json.dumps(data,indent=2)+'\n')


def main():
    now=datetime.now(timezone.utc).isoformat();rental=BASE/'rental-03'
    p=read(BASE/'protocol.json');result=read(rental/'remote/run-01/result.json')
    replay=read(rental/'local-verification.json')
    assert replay['evidence_valid']
    spec=importlib.util.spec_from_file_location('frozen_layout_analysis',BASE/'source/analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    decision=module.analyze(result,p)
    assert decision==replay['decision'] and not decision['publication_goal_achieved']
    assert len(result['products'])==replay['products_replayed']
    assert len(result['solves'])+len(result['warmups'])==replay['states_replayed']
    assert read(rental/'destruction.json')['success']
    assert read(rental/'monitor.json')['state']=='collected_and_destroyed'
    assert not read(rental/'post-destruction-instances.json')['instances']
    assert sha(rental/'evidence.tar.gz')==read(rental/'archive-receipt.json')['sha256']
    for name,digest in p['package_source_pins'].items():
        for directory in [BASE/'source',rental/'remote/source',ROOT/'rescope/hex_modal_layout_v2']:
            assert sha(directory/name)==digest,(directory,name)
    assert sha(BASE/'package-02.tar')==read(BASE/'package-receipt.json')['sha256']
    with (ROOT/'ledger.csv').open() as f:ledger=list(csv.DictReader(f))
    with (REPORT/'before-outcome-ledger.csv').open() as f:previous=list(csv.DictReader(f))
    assert ledger[:-1]==previous and len(ledger)==15
    cost=read(BASE/'cost-estimate.json')
    assert Decimal(ledger[-1]['rate_time_estimate_usd'])==Decimal(cost['rate_time_estimate_usd'])
    section=(ROOT/'scientific_report.md').read_text().split('## 42. ',1)[1]
    assert f"{float(cost['rate_time_estimate_usd']):.8f}" in section
    assert str(replay['products_replayed']) in section and str(replay['states_replayed']) in section
    assert 'advance_to_charged_campaign=false' in section
    for row in decision['comparisons']:
        if row['primary']:
            assert f"{row['paired_median_ratio']:.6f}" in section
    prior=read(REPORT/'prior-seals-initial.json')['manifests'];checked=0
    for item in prior:
        path=ROOT/item['manifest'];assert sha(path)==item['sha256']
        for name,expected in read(path)['files'].items():
            f=ROOT/name;assert f.stat().st_size==expected['bytes'] and sha(f)==expected['sha256'],name
            checked+=1
    write(REPORT/'prior-seals-final.json',{'utc':now,'passed':True,'entries_verified':checked,'manifests':prior})
    links=0
    for name in ['scientific_report.md','research_process_review.md','invoice_reconciliation.md']:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',(ROOT/name).read_text()):
            if target.startswith(('https://','http://','#','mailto:')):continue
            target=target.split('#',1)[0]
            if target:assert (ROOT/target).exists(),(name,target);links+=1
    for name in ['scientific_report.md','research_process_review.md','logbook.md','ledger.csv','invoice_reconciliation.md']:
        destination=REPORT/('completed-'+name);assert not destination.exists();shutil.copyfile(ROOT/name,destination)
    for name in ['works.tsv','catalog.json','references.bib','INDEX.md','reading-notes.json','update_catalog.py']:
        destination=REPORT/('bibliography-'+name);assert not destination.exists();shutil.copyfile(ROOT/'biblio'/name,destination)
    write(REPORT/'audit.json',{'utc':now,'passed':True,'status':result['status'],
        'products_replayed':replay['products_replayed'],'states_replayed':replay['states_replayed'],
        'timing_rows':len(result['timings']),'development_gates':decision['development_gates'],
        'source_pins_checked':len(p['package_source_pins']),'prior_sealed_entries_checked':checked,
        'local_links_checked':links,'billing_rows':len(ledger),'instance_destroyed':True,
        'publication_goal_achieved':False,'old_profile_gate_unchanged':True})
    write(REPORT/'revision.json',{'utc':now,'section':42,
        'immediately_previous_report_sha256':sha(REPORT/'before-outcome-scientific_report.md'),
        'completed_report_sha256':sha(REPORT/'completed-scientific_report.md')})
    secret=Path('/tmp/jpdc-vast-20260922/api-key').read_bytes().strip()
    for target,directories in [(BASE/'MANIFEST.json',[BASE,ROOT/'rescope/hex_modal_layout_v2']),
                               (REPORT/'MANIFEST.json',[REPORT])]:
        files={}
        for directory in directories:
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or path==target or '__pycache__' in path.parts:continue
                if path.suffix not in {'.gz','.tar','.npz','.so','.dylib','.png'}:
                    assert secret not in path.read_bytes(),f'Credential found in {path}'
                files[str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':sha(path)}
        write(target,{'sealed_utc':now,'scope':'Fixed-work layout and symmetric-storage development; all failures retained','files':files})
        print(json.dumps({'manifest':str(target.relative_to(ROOT)),'files':len(files),'sha256':sha(target)}))


if __name__=='__main__':main()
