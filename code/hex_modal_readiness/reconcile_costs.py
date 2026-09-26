"""Reconstruct the missing rental ledger from retained records, without provider access.

Unknown charges stay unknown. Account-credit deltas are never added as another
charge. The generated subtotal is expressly incomplete, not a settled invoice.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-ledger', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {}

    def load(path):
        path = ROOT / path
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw, parse_float=Decimal)

    with args.previous_ledger.open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames)
    assert [r['session_id'] for r in rows] == ['ALL-MONTH1-RENTALS', 'A2-CUDA-ACCEPTANCE']
    fields += ['rate_time_hours', 'rate_time_estimate_usd', 'rate_scope',
               'known_amount_subtotal_usd', 'evidence']
    for row in rows:
        row['known_amount_subtotal_usd'] = row['cumulative_usd']
    subtotal = Decimal(rows[-1]['cumulative_usd'])
    amount = lambda x: format(x.quantize(Decimal('.000000001')), 'f')
    entries = []

    def append(session, date, gpu, host, evidence, rate=None, hours=None,
               estimate=None, scope='unresolved', note='', compute_rate=None):
        nonlocal subtotal
        row = {k: 'unresolved' for k in fields}
        row.update(date=date, session_id=session, provider='vast.ai', host_id=host,
                   gpu_model=gpu, gpu_variant='unresolved',
                   status='cost-unresolved' if estimate is None else 'estimated-pending-invoice',
                   notes=note, rate_scope=scope, evidence='; '.join(evidence))
        if estimate is not None:
            subtotal += estimate
            row.update(hourly_rate_usd=amount(rate), rate_time_hours=amount(hours),
                       rate_time_estimate_usd=amount(estimate))
            if compute_rate is not None:
                row.update(compute_usd=amount(compute_rate * hours),
                           storage_usd=amount((rate - compute_rate) * hours))
        row['known_amount_subtotal_usd'] = amount(subtotal)
        # Two earlier rescope charges and the ancillary charges are unresolved.
        row['cumulative_usd'] = 'unresolved'
        rows.append(row)
        entries.append({'session_id': session, 'rate_time_estimate_usd':
                        None if estimate is None else str(estimate), 'evidence': evidence})

    calibration = 'results/rescope/gpu-calibration-20260922/session.json'
    load(calibration)
    append('RESCOPE-CALIBRATION-20260922', '2026-09-22', 'RTX 3090', 'unresolved',
           [calibration], note='Retained session explicitly reports unknown rental cost; no rate, '
           'billing duration or invoice inferred. Distinct from month-1 rentals.')
    matched = 'results/rescope/matched-gpu-20260922/rental-stop.json'
    stopped = load(matched)
    append('RESCOPE-MATCHED-20260922', '2026-09-22', 'RTX 3090', str(stopped['instance_id']),
           [matched], note='Stop receipt retains disk and warns that storage remains billed. '
           'No supported price or billed duration; later empty inventories do not establish '
           'this instance\'s individual storage charge.')
    for name, cost_name, duration_key, estimate_key in [
        ('matched-convergence-gpu-20260922', 'cost-observation.json',
         'elapsed_allocation_seconds', 'rate_times_elapsed_estimate_usd'),
        ('matched-symmetry-20260923', 'cost-observation.json',
         'elapsed_allocation_seconds', 'rate_times_elapsed_estimate_usd'),
        ('matched-symmetry-20260923/rental-attempt-02', 'cost.json',
         'rental_wall_seconds', 'rate_time_estimate_usd'),
        ('matched-symmetry-20260923/rental-attempt-03', 'cost.json',
         'rental_wall_seconds', 'rate_time_estimate_usd')]:
        root = 'results/rescope/' + name
        rental, cost = load(root + '/rental.json'), load(root + '/' + cost_name)
        destruction = load(root + '/destruction.json')
        assert destruction['success'] and destruction['instance_id'] == rental['instance_id']
        hours = cost[duration_key] / 3600
        estimate, rate = cost[estimate_key], rental['rate_usd_per_hour']
        assert abs(hours * rate - estimate) < Decimal('1e-8')
        append('VAST-' + str(rental['instance_id']), rental['created_utc'][:10], 'RTX 4090',
               str(rental['instance_id']), [root + '/rental.json', root + '/' + cost_name,
                                          root + '/destruction.json'], rate, hours, estimate,
               'quoted compute plus allocated storage',
               'Rate times allocation wall time; can include stopped/setup intervals. '
               'Transfers, tax and settlement unresolved. Account-credit decrease is a '
               'separate observation, not another charge.')
    root = 'results/rescope/published-optimizer-20260923/rental-01'
    rental = load(root + '/rental.json')
    cost = load(root + '/cleanup-verification.json')
    assert cost['absent_from_instance_list'] and cost['instance_id'] == rental['instance_id']
    assert abs(cost['elapsed_hours'] * cost['listed_offer_rate_usd_per_hour'] -
               cost['listed_rate_times_elapsed_usd']) < Decimal('1e-8')
    append('VAST-' + str(rental['instance_id']), rental['created_utc'][:10], 'RTX 4090',
           str(rental['instance_id']), [root + '/rental.json', root + '/cleanup-verification.json'],
           cost['listed_offer_rate_usd_per_hour'], cost['elapsed_hours'],
           cost['listed_rate_times_elapsed_usd'], 'listed offer total; disk allocation caveat',
           'Preserved cleanup estimate; listed offer may differ from requested disk allocation. '
           'Transfers, tax and settlement unresolved; account-credit change is not added.')
    for name in ['hex-modal-cuda-20260923', 'hex-modal-replication-20260923']:
        root = 'results/rescope/' + name + '/rental-01'
        rental, cost = load(root + '/rental.json'), load(root + '/cost.json')
        destruction = load(root + '/destruction.json')
        assert destruction['success'] and destruction['instance_id'] == rental['instance_id']
        rate = cost['compute_and_80gb_storage_usd_per_hour']
        hours = cost['rental_seconds'] / 3600
        estimate = cost['rate_times_duration_usd_excluding_transfers']
        assert abs(rate * hours - estimate) < Decimal('1e-8')
        append('VAST-' + str(rental['instance_id']), rental['created_utc'][:10],
               rental['offer']['gpu_name'], str(rental['instance_id']),
               [root + '/rental.json', root + '/cost.json', root + '/destruction.json'],
               rate, hours, estimate, 'compute plus 80 GB storage',
               'Compute and storage columns are rate-time estimates, not invoiced components. '
               'Transfers, tax and final settlement unresolved; no credit-delta double counting.',
               compute_rate=cost['compute_usd_per_hour'])
    with (out / 'ledger.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'previous_ledger_sha256': hashlib.sha256(args.previous_ledger.read_bytes()).hexdigest(),
               'previous_rows_preserved': 2, 'rescope_sessions_added': len(entries),
               'priced_rescope_sessions': sum(e['rate_time_estimate_usd'] is not None for e in entries),
               'unpriced_rescope_sessions': sum(e['rate_time_estimate_usd'] is None for e in entries),
               'known_rescope_rate_time_estimates_usd': str(subtotal - Decimal('10.27')),
               'known_amount_subtotal_usd': str(subtotal), 'complete_cumulative_cost_usd': None,
               'settled_invoice': False, 'new_rentals': 0, 'new_spending_usd': 0,
               'account_credit_deltas_added_to_spending': False,
               'limitations': 'Two unpriced rescope sessions, transfer/tax/settlement gaps and '
                              'the historical aggregate/estimate basis remain unresolved.',
               'entries': entries, 'source_pins': sources}
    (out / 'reconciliation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (out / 'reconcile_costs.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: receipt[k] for k in ['rescope_sessions_added', 'priced_rescope_sessions',
                     'unpriced_rescope_sessions', 'known_amount_subtotal_usd']}, indent=2))


if __name__ == '__main__':
    main()
