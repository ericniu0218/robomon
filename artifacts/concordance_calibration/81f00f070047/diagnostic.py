"""Check primary concordance interval coverage with paired conditions and timeout censoring."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import ndtr

from robomon.analysis.landmarks import bootstrap_concordance
from robomon.utils.io import file_sha256,source_hash,stage_dir,write_json,write_manifest


def sample(effect,seed):
    rng=np.random.default_rng(seed)
    rows=[]
    for task in range(12):
        for unit in range(16):
            noise=rng.normal()
            outcome_uniform=rng.random()
            for condition in range(2):
                failure=int(outcome_uniform<(.25+.3*condition))
                score=effect*failure+noise
                for landmark in range(1,6 if failure else 4):
                    rows.append(dict(episode_id=f'{task}/{unit}/{condition}',pair_id=f'{task}/{unit}',
                                     task=task,condition=str(condition),landmark=landmark,failure=failure,
                                     score=score,elapsed=float(landmark)))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=int,default=48)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--seed-start',type=int,default=0)
    args=parser.parse_args()
    config={'stage':'concordance_calibration','source_hash':source_hash(),
            'diagnostic_sha':file_sha256(Path(__file__)),'datasets':args.datasets,
            'n_boot':args.bootstrap,'effects':[0.,1.], 'tasks':12,'pair_units_per_task':16,
            'seed_start':args.seed_start,
            'noise':'one standard normal shared by all landmarks/conditions of each unit',
            'censoring':'failures have five landmarks; successes three'}
    output=stage_dir('concordance_calibration',config)
    write_manifest(output,'concordance_calibration',config)
    print(f'OUTPUT {output}',flush=True)
    records=[]
    for effect in config['effects']:
        truth=float(ndtr(effect/np.sqrt(2)))
        for seed in range(args.seed_start,args.seed_start+args.datasets):
            data=sample(effect,seed)
            result=bootstrap_concordance(data,'score',n_boot=args.bootstrap,seed=1000+seed,cluster='pair_id')
            control=bootstrap_concordance(data,'elapsed',n_boot=1,cluster='pair_id')
            if control['estimate']!=.5:
                raise AssertionError('Elapsed-time negative control failed')
            records.append({'effect':effect,'true_concordance':truth,'seed':seed,**result})
            print(f'{len(records)}/{2*args.datasets} effect={effect} seed={seed}',flush=True)
    summaries=[]
    for effect in config['effects']:
        rows=[r for r in records if r['effect']==effect]
        summaries.append({'effect':effect,'truth':rows[0]['true_concordance'],
                          'mean':float(np.mean([r['estimate'] for r in rows])),
                          'coverage':float(np.mean([r['ci_low']<=r['true_concordance']<=r['ci_high'] for r in rows]))})
    write_json({'scope':'synthetic calibration only','summaries':summaries,'datasets':records},output/'results.json')
    print(summaries,flush=True)


if __name__=='__main__':
    main()
