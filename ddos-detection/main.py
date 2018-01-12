import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from datetime import datetime
from utils import save_results, get_classifier, get_file_num, \
        pickle_summarized_data, get_saved_data, get_binetflow_files, \
        get_feature_labels, to_tf_label, get_start_time_for, TIME_FORMAT
import tensorflow as tf
from summarizer import Summarizer
import numpy as np


def get_base_name(filename):
    return filename.split('/')[-1].split('.')[0]


def aggregate_file(interval, file_name, start=None):
    """ Aggregate the data within the windows of time

        interval:       time in seconds to aggregate data
        file_name:      which file to record
        start:          start time to record data, if none given then it starts
                        from te beginning.

        returns: array of the aggregated data in each interval
    """
    if start is None:
        start = get_start_time_for(file_name)

    start = datetime.strptime(start, TIME_FORMAT)
    summaries = [Summarizer() for _ in range(10)]
    self.total = 0
    with open(file_name, 'r+') as data:
        headers = data.readline().strip().lower().split(',')
        for line in data:
            self.total += 1
            args = line.strip().split(',')
            time = datetime.strptime(args[0], TIME_FORMAT)
            window = int((time - start).total_seconds() / interval)
            if window < 0:
                continue
            if window >= len(summaries):
                for i in range(window + 1):
                    summaries.append(Summarizer())
            item = dict(zip(headers, args))
            if 'Background' in item['label']:
                continue
            summaries[window].add(item)

    summaries = [s for s in summaries if s.used]
    print('Got {}/{}'.format(len(summaries), total))
    basename = get_base_name(file_name)
    with open('aggregated_binetflows/{}'.format(basename), 'w+') as out:
        for summary in summaries:
            out.write(','.join(summary.get_feature_list()) + '\n')


def aggregate_and_pickle(interval, file_name, start=None):
    summary = aggregate_file(interval, file_name, start)
    pickle_summarized_data(interval, file_name, summary)
    return summary


if __name__ == '__main__':
    all_intervals = [.5, 1, 2, 5]

    binet_files = [
        'binetflows/capture20110815.binetflow',
        'binetflows/capture20110818.binetflow',
        'binetflows/capture20110818-2'
    ]

    for binet in binet_files:
        aggregate_file(1, binet)

    # Avoid error in keras
    import gc
    gc.collect()
