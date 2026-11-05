"""Compare freshly retrained models with the immutable pre-fix reference models."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

import joblib
import numpy as np
import pandas as pd
import torch
from air_prediction.predict import predict_latest


def verify():
    references=ROOT/'artifacts/phase2_external/reference_models'
    active=ROOT/'artifacts/phase2_research'
    original=json.loads((references/'training_manifest.json').read_text())
    updated=json.loads((active/'run_manifest.json').read_text())
    assert original['status']==updated['status']=='complete'
    result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
            'reference_source_sha256':original['source_code_sha256'],
            'active_source_sha256':updated['source_code_sha256'],'models':{}}
    for name in ('lstm','bilstm','hybrid'):
        old_path,new_path=references/f'{name}.joblib',active/f'{name}.joblib'
        assert sha256(old_path.read_bytes()).hexdigest()==original['model_sha256'][name]
        assert sha256(new_path.read_bytes()).hexdigest()==updated['model_sha256'][name]
        old,new=joblib.load(old_path),joblib.load(new_path)
        assert old['config']==new['config'] and old['columns']==new['columns']
        before=old['pipeline'].dl if name=='hybrid' else old['pipeline']
        after=new['pipeline'].dl if name=='hybrid' else new['pipeline']
        assert before.config==after.config
        for field in ('feature_means','feature_stds','loss_history_'):
            np.testing.assert_array_equal(getattr(before,field),getattr(after,field))
        np.testing.assert_array_equal(before.feature_medians_,after.feature_medians_)
        pd.testing.assert_index_equal(before.training_origins_,after.training_origins_)
        for field in ('category_levels_','target_mean_','target_std_','training_rows_',
                      'preprocessing_fit_start_','preprocessing_fit_end_'):
            assert getattr(before,field)==getattr(after,field),field
        digest=sha256()
        old_state,new_state=before.model.state_dict(),after.model.state_dict()
        assert old_state.keys()==new_state.keys()
        for field in old_state:
            assert torch.equal(old_state[field],new_state[field]),field
            digest.update(field.encode())
            digest.update(old_state[field].cpu().numpy().tobytes())
        if name=='hybrid':
            assert old['pipeline'].correction_weight==new['pipeline'].correction_weight
            old_head,new_head=old['pipeline'].tabular_model,new['pipeline'].tabular_model
            assert joblib.hash(old_head.named_steps['preprocess'])==joblib.hash(new_head.named_steps['preprocess'])
            assert old_head.named_steps['regressor'].get_booster().save_raw()==new_head.named_steps['regressor'].get_booster().save_raw()
        csv=ROOT/'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
        old_prediction=predict_latest(old_path,csv)['pm25_ug_m3']
        new_prediction=predict_latest(new_path,csv)['pm25_ug_m3']
        assert old_prediction==new_prediction
        result['models'][name]={'exact_neural_parameter_sha256':digest.hexdigest(),
            'preprocessing_and_target_scaling_identical':True,'reference_model_sha256':original['model_sha256'][name],
            'active_model_sha256':updated['model_sha256'][name],'latest_forecast':new_prediction}
    result['checks']=['independent fresh training','exact neural state tensors','training origins and preprocessing',
                      'target statistics and loss histories','exact hybrid tree bytes and preprocessing',
                      'identical saved-model forecasts']
    for directory in (active,ROOT/'docs/results/phase2_research'):
        (directory/'portability_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    verify()
