""" Take each file and aggregate the files by a certain time frame.

Each packet must be looked at as a group. So aggregate all the feateres that
lie close together by some pre-determined time range. This output file will
be used for machine learning.
"""
import os
from datetime import datetime, timedelta


PCAP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

def write_aggregated_file(filename):
    features = []
    basename = filename.split('.')[0].split('/')[1]
    with open(filename, 'r+') as f:
        with open('aggregated_pcap/'+basename + '.featureset.csv',
                'w+') as out:
            headers = f.readline().strip().split(',')
            last = None
            bound = None
            current = list(range(4))
            feats = [
                'packet_length',
                'fragment_offset',
                'TTL',
                'proto'
            ]
            is_botnet = False
            for line in f:
                info = line.strip().split(',')
                features = dict(zip(headers, info))
                time = datetime.strptime(features['start_time'], PCAP_FORMAT)

                if last is None or time > bound:
                    last = time
                    bound = last + timedelta(seconds=.15)
                    out.write(','.join([str(x) for x in current]
                        ) + ',{}\n'.format(
                        'Botnet' if is_botnet else 'Normal'))
                    current = [0, 0, 0, 0, 0, 0]
                    is_botnet = False
                    # Write the features to out.
                elif last < time < bound:
                    # Fill the features in.
                    proto = int(features['proto'])
                    if proto == 1:
                        current[0] += 1
                    elif proto == 6:
                        current[1] += 1
                    elif proto == 17:
                        current[2] += 1

                    current[3] += int(features['packet_length'])
                    current[4] += int(features['fragment_offset'])
                    current[5] += int(features['TTL'])
                    if features['label'] == 'Botnet':
                        is_botnet = True
            print('finished', filename)


if __name__ == '__main__':
    data = os.listdir('pcap_feature_label')
    data = ['pcap_feature_label/' + x for x in data]
    for d in data:
        write_aggregated_file(d)
