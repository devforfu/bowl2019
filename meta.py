from collections import OrderedDict
import utils as U


def compute_meta_data(dataset, *datasets):
    datasets = [dataset] + list(datasets)
    uniq = OrderedDict()
    uniq['title_event_code'] = U.unique(datasets, column='title_event_code')
    uniq['title'] = U.unique(datasets, column='title')
    uniq['event_code'] = U.unique(datasets, column='event_code')
    uniq['event_id'] = U.unique(datasets, column='event_id')
    uniq['world'] = U.unique(datasets, column='world')
    uniq['type'] = U.unique(datasets, column='type')
    uniq['title_world'] = U.unique(datasets, column='title_world')
    uniq['title_type'] = U.unique(datasets, column='title_type')
    uniq['world_type'] = U.unique(datasets, column='world_type')
    asm_datasets = [ds.query('type == "Assessment"') for ds in datasets]
    uniq['assessment_titles'] = U.unique(asm_datasets, column='title')
    win_codes = {t: 4100 for t in uniq['title']}
    win_codes['Bird Measurer (Assessment)'] = 4110
    ref_ts = dataset['timestamp'].min()
    meta = {'win_codes': win_codes, 'ref_ts': ref_ts, **uniq}
    return U.named_tuple('Meta', **meta)