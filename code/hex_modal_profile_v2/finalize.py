"""Audit printed results, preserve prior seals, and seal the completed diagnostic."""
import csv
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/rescope/hex-modal-profile-v2-20260923'
REPORT=ROOT/'results/reports/scientific-report-hex-modal-profile-v2-20260923'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(p.read_text())


def write(p,v):
    assert not p.exists(),p
    p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    now=datetime.now(timezone.utc).isoformat()
    rental=BASE/'rental-01'
    result=read(rental/'remote/run-01/result.json')
    replay=read(rental/'local-verification.json')
    p=read(BASE/'protocol.json')
    assert result['status']=='completed' and len(result['solves'])==70
    assert len(result['warmups'])==20 and replay['states_replayed']==90
    assert all(x['physical_passed'] for x in result['solves'])
    assert all(not x['physical_passed'] and x['expected_rejection'] for x in result['warmups'])
    assert len(replay['cross_path_checks'])==70 and all(x['valid'] for x in replay['cross_path_checks'])
    spec=importlib.util.spec_from_file_location('sealed_analysis',BASE/'source/analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    decision=module.analyze(result,p)
    assert decision==replay['diagnostic_decision']
    primary=[x for x in decision['summaries'] if x['primary']]
    assert sum(x['profile_qualified'] for x in primary)==8
    assert not decision['advance_to_charged_campaign']
    assert all(x['passes_planning_ratio'] for x in decision['comparisons'] if x['primary'])
    report=(ROOT/'scientific_report.md').read_text();section=report.split('## 40. ',1)[1]
    for x in primary:
        assert f"{x['median_plain_seconds']:.6f}" in section
    assert f"{replay['maximum_normal_true_residual']:.10e}" in section
    assert 'advance_to_charged_campaign=false' in section
    assert read(rental/'destruction.json')['success']
    assert read(rental/'monitor.json')['state']=='collected_and_destroyed'
    assert not read(rental/'post-destruction-instances.json')['instances']
    assert read(rental/'collection-verification.json')['files']==154
    assert sha(rental/'evidence.tar.gz')==read(rental/'archive-receipt.json')['sha256']
    for name,digest in p['package_source_pins'].items():
        for directory in [BASE/'source',rental/'remote/source',ROOT/'rescope/hex_modal_profile_v2']:
            assert sha(directory/name)==digest,(directory,name)
    with (ROOT/'ledger.csv').open() as f:ledger=list(csv.DictReader(f))
    with (REPORT/'before-outcome-ledger.csv').open() as f:previous=list(csv.DictReader(f))
    assert ledger[:-1]==previous and len(ledger)==13
    cost=read(BASE/'cost-estimate.json')
    assert Decimal(ledger[-1]['rate_time_estimate_usd'])==Decimal(cost['rate_time_estimate_usd'])
    assert f"{float(cost['rate_time_estimate_usd']):.8f}" in section
    assert f"{float(cost['known_amount_subtotal_usd']):.8f}" in section
    counts=read(ROOT/'biblio/catalog.json')['counts']
    assert counts=={'works':98,'works_with_local_full_text':80,'metadata_only_works':18,
                   'pdf_files':81,'postscript_files':1,'html_files':1}
    prior=read(REPORT/'prior-seals-initial.json')['manifests']
    for name in ['results/rescope/hex-central-parity-20260923/MANIFEST.json',
                 'results/reports/scientific-report-hex-central-parity-20260923/MANIFEST.json']:
        prior.append({'manifest':name,'sha256':sha(ROOT/name),'files':len(read(ROOT/name)['files'])})
    checked=0
    for item in prior:
        path=ROOT/item['manifest'];assert sha(path)==item['sha256']
        for name,expected in read(path)['files'].items():
            file=ROOT/name;assert file.stat().st_size==expected['bytes'] and sha(file)==expected['sha256'],name
            checked+=1
    write(REPORT/'prior-seals-final.json',{'utc':now,'passed':True,'entries_verified':checked,'manifests':prior})
    links=0
    for name in ['scientific_report.md','research_process_review.md','invoice_reconciliation.md']:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',(ROOT/name).read_text()):
            if target.startswith(('https://','http://','#','mailto:')):continue
            target=target.split('#',1)[0]
            if target:assert (ROOT/target).exists(),(name,target);links+=1
    for name in ['scientific_report.md','research_process_review.md','logbook.md',
                 'invoice_reconciliation.md','ledger.csv']:
        dest=REPORT/('completed-'+name);assert not dest.exists();shutil.copyfile(ROOT/name,dest)
    for name in ['works.tsv','catalog.json','references.bib','INDEX.md','reading-notes.json','update_catalog.py']:
        dest=REPORT/('bibliography-'+name);assert not dest.exists();shutil.copyfile(ROOT/'biblio'/name,dest)
    write(REPORT/'audit.json',{'utc':now,'passed':True,'normal_states':70,'capped_states':20,
        'cross_path_checks':70,'primary_median_rows_recomputed':10,'qualified_primary_profiles':8,
        'planning_gate_passed':False,'source_pins_checked':len(p['package_source_pins']),
        'prior_sealed_entries_checked':checked,'local_links_checked':links,'billing_rows':13,
        'instance_destroyed':True,'publication_claim':False})
    write(REPORT/'revision.json',{'utc':now,'section':40,
        'immediately_previous_report_sha256':sha(REPORT/'before-outcome-scientific_report.md'),
        'completed_report_sha256':sha(REPORT/'completed-scientific_report.md')})
    secret=Path('/tmp/jpdc-vast-20260922/api-key').read_bytes().strip()
    for target,directories in [(BASE/'MANIFEST.json',[BASE,ROOT/'rescope/hex_modal_profile_v2']),
                               (REPORT/'MANIFEST.json',[REPORT])]:
        files={}
        for directory in directories:
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or path==target or '__pycache__' in path.parts:continue
                # Every source and textual receipt is checked without printing the credential.
                if path.suffix not in {'.gz','.tar','.npz','.so','.dylib','.png'}:
                    assert secret not in path.read_bytes(),f'Credential found in {path}'
                files[str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':sha(path)}
        write(target,{'sealed_utc':now,'scope':'Complete warm diagnostic; failed planning gate retained','files':files})
        print(json.dumps({'manifest':str(target.relative_to(ROOT)),'files':len(files),'sha256':sha(target)}))


if __name__=='__main__':main()
