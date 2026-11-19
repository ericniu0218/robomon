"""Finite simulation check of composite-logistic bias and clustered interval coverage.

This is an estimator diagnostic under a declared synthetic model, not a guarantee of coverage for
robot data. Features/labels are repeated across landmarks to test episode-level dependence.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit

from robomon.analysis.conditional import fit_pairwise_conditional
from robomon.analysis.development import json_safe
from robomon.utils.io import file_sha256,source_hash,stage_dir,write_json,write_manifest


def synthetic(beta,seed,paired=False):
    rng=np.random.default_rng(seed)
    rows=[]
    for task in range(12):
        for episode in range(16 if paired else 32):
            base=rng.normal(size=3)
            shared_uniform=rng.random()
            for condition in range(2 if paired else 1):
                support,dispersion,consistency=base+(.3*rng.normal(size=3) if paired else 0)
                failure=int(shared_uniform<expit((task-6)/6+.5*condition+beta*support+.5*dispersion-.2*consistency))
                for landmark in (25,50,75):
                    rows.append(dict(task=task,condition=str(condition),episode_id=f'{task}/{episode}/{condition}',
                                     pair_id=f'{task}/{episode}',failure=failure,landmark=landmark,
                                     support=support,dispersion=dispersion,consistency=consistency))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=int,default=24)
    parser.add_argument('--bootstrap',type=int,default=200)
    parser.add_argument('--seed-start',type=int,default=0)
    parser.add_argument('--paired-conditions',action='store_true')
    args=parser.parse_args()
    config={'stage':'conditional_calibration','source_hash':source_hash(),
            'diagnostic_sha':file_sha256(Path(__file__)),'datasets':args.datasets,
            'n_boot':args.bootstrap,'true_support_coefficients':[0.,.8],
            'tasks':12,'episodes_per_task':32,'landmarks':3,'feature_scales':[1.,1.,1.],
            'seed_start':args.seed_start,'paired_conditions':args.paired_conditions,'interval':'basic'}
    output=stage_dir('conditional_calibration',config)
    write_manifest(output,'conditional_calibration',config)
    print(f'OUTPUT {output}',flush=True)
    records=[]
    for beta in config['true_support_coefficients']:
        for seed in range(args.seed_start,args.seed_start+args.datasets):
            result=fit_pairwise_conditional(synthetic(beta,seed,args.paired_conditions),('support','dispersion','consistency'),
                    n_boot=args.bootstrap,seed=1000+seed,scales={'support':1.,'dispersion':1.,'consistency':1.})
            effect=(result.get('coefficients') or {}).get('support')
            records.append({'true_coef':beta,'dataset_seed':seed,'status':result['status'],
                            'effect':effect,'bootstrap_valid':result.get('bootstrap_valid',0)})
            print(f'{len(records)}/{2*args.datasets} beta={beta} seed={seed}',flush=True)
    summaries=[]
    for beta in config['true_support_coefficients']:
        valid=[r['effect'] for r in records if r['true_coef']==beta and r['status']=='ok']
        summaries.append({'true_coef':beta,'valid_datasets':len(valid),
                          'mean_coef':float(np.mean([r['coef'] for r in valid])) if valid else None,
                          'coverage':float(np.mean([r['ci_low']<=beta<=r['ci_high'] for r in valid])) if valid else None})
    write_json(json_safe({'scope':'synthetic calibration only','summaries':summaries,'datasets':records}),
               output/'results.json')
    print(summaries,flush=True)


if __name__=='__main__':
    main()
