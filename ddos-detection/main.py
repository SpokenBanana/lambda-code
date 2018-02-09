import os
from absl import app
from absl import flags
from datetime import datetime
from utils import save_results, get_classifier, get_file_num, \
        pickle_summarized_data, get_saved_data, get_binetflow_files, \
        get_feature_labels, to_tf_label, get_start_time_for, TIME_FORMAT, \
        get_feature_order
from summarizer import Summarizer


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'attack_type', None, 'Type of files to aggregate together.')
flags.DEFINE_float(
    'interval', None, 'Interval in seconds to aggregate connections.')


def get_base_name(filename):
    return filename.split('/')[-1].split('.')[0]


def aggregate_file(interval, file_name, output_name, start=None):
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
    summaries = {}
    total = 0
    botnets = 0
    background = 0
    normal = 0
    print('starting', file_name)
    with open(file_name, 'r+') as data:
        headers = data.readline().strip().lower().split(',')
        for line in data:
            total += 1
            args = line.strip().split(',')
            time = datetime.strptime(args[0], TIME_FORMAT)
            window = int((time - start).total_seconds() / interval)
            # if window < 0:
            #     continue
            # if window >= len(summaries):
            #     for i in range(window + 1):
            #         summaries.append(Summarizer())
            item = dict(zip(headers, args))
            if 'Background' in item['label']:
                background += 1
                continue
            elif 'Normal' in item['label']:
                normal += 1
            elif 'Botnet' in item['label']:
                botnets += 1

            if window not in summaries:
                summaries[window] = Summarizer()
            summaries[window].add(item)

    summaries = [s for s in summaries.values() if s.used]
    total_connections = sum(s.data['n_conn'] for s in summaries)
    print('Average Connection per interval: {}'.format(
        total_connections / len(summaries)))
    print('Got {}/{}'.format(len(summaries), total))
    print('{} botnets, {} normal, {} background'.format(botnets, normal,
          background))
    append_to_ddos_featureset(summaries, output_name)

    # Use this if you want a featureset for each file.

    # basename = get_base_name(file_name)
    # filename = 'minute_aggregated/{}.aggregated.csv'.format(basename)
    # write_featureset(filename, summaries)


def append_to_ddos_featureset(summaries, output_name):
    with open(output_name, 'a') as out:
        for summary in summaries:
            out.write(','.join(summary.get_feature_list()) + '\n')


def write_featureset(filename, summaries):
    with open(filename, 'w+') as out:
        out.write(','.join(get_feature_order()) + ',label\n')
        for summary in summaries:
            out.write(','.join(summary.get_feature_list()) + '\n')


def main(_):
    binet_files = [
        'binetflows/capture20110815.binetflow',
        'binetflows/capture20110818.binetflow',
        'binetflows/capture20110818-2.binetflow'
    ]

    # 1,2,5,9,13
    spam_files = [
        'binetflows/capture20110811.binetflow',
        'binetflows/capture20110812.binetflow',

        'binetflows/capture20110815-2.binetflow',

        'binetflows/capture20110817.binetflow',

        'binetflows/capture20110815-3.binetflow',
    ]

    # 1-4 and 9-11
    irc_files = [
        'binetflows/capture20110810.binetflow',
        'binetflows/capture20110811.binetflow',
        'binetflows/capture20110812.binetflow',
        'binetflows/capture20110815.binetflow',

        'binetflows/capture20110817.binetflow',
        'binetflows/capture20110818.binetflow',
        'binetflows/capture20110818-2.binetflow',
    ]

    p2p_files = ['binetflows/capture20110819.binetflow']

    # Set up the file that holds all this information.
    output_name = 'minute_aggregated/{}-{}s.featureset.csv'.format(
        FLAGS.attack_type, FLAGS.interval)
    with open(output_name, 'w+') as out:
        out.write(','.join(Summarizer().features) + ',label\n')

    attack_files = []
    if FLAGS.attack_type == 'ddos':
        attack_files = binet_files
    elif FLAGS.attack_type == 'spam':
        attack_files = spam_files
    elif FLAGS.attack_type == 'irc':
        attack_files = irc_files
    elif FLAGS.attack_type == 'p2p':
        attack_files = p2p_files
    elif FLAGS.attack_type == 'general':
        attack_files = ['binetflows/{}'.format(binet) for binet in os.listdir('binetflows')]

    for binet in attack_files:
        aggregate_file(FLAGS.interval, binet, output_name)

    # Avoid error in keras
    import gc
    gc.collect()


if __name__ == '__main__':
    app.run(main)
