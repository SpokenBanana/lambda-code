""" Take each file and aggregate the files by a certain time frame.

Each packet must be looked at as a group. So aggregate all the feateres that
lie close together by some pre-determined time range. This output file will
be used for machine learning.
"""
import os
from datetime import datetime, timedelta
from collections import Counter
import numpy


PCAP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def entropy(ctr):
    a = ctr.values()
    b = sum(a)
    al = numpy.asarray(a)
    al = al / float(b)
    return -sum(numpy.log(al) * al)


class Summary:

    def __init__(self):
        self.average_packet = 0
        self.packet_count = 0

        self.average_ttl = 0
        self.ttl_count = 0

        self.average_fragment_count = 0

        self.n_udp = 0
        self.n_tcp = 0
        self.n_icmp = 0
        self.label = False

        self.ws_counter = Counter()
        self.hs_counter = Counter()
        self.pnd_counter = Counter()
        self.pns_counter = Counter()
        self.tcp_flag_counter = Counter()

    def add(self, features):
        self.packet_count += 1
        self.average_packet += (
            features['packet_length'] -
            self.average_packet) / self.packet_count

        self.ttl_count += 1
        self.average_ttl += (
            features['TTL'] - self.average_ttl) / self.packet_count

        self.average_fragment_count += 1

        self.ws_counter[features['ws']] += 1
        self.hs_counter[features['hs']] += 1
        self.pns_counter[features['pns']] += 1
        self.pnd_counter[features['pnd']] += 1
        self.tcp_flag_counter[features['tcp_flag']] += 1

        proto = int(features['proto'])

        if proto == 1:
            self.n_udp += 1
        elif proto == 6:
            self.n_tcp += 1
        elif proto == 17:
            self.n_icmp += 1

        if features['label'] == 'Botnet':
            self.label = True

    def to_feature_array(self):
        return [
            self.n_udp,
            self.n_tcp,
            self.n_icmp,
            self.average_packet,
            self.average_ttl,
            self.average_fragment_count,
            entropy(self.ws_counter),
            entropy(self.hs_counter),
            entropy(self.pnd_counter),
            entropy(self.pns_counter),
            entropy(self.tcp_flag_counter),
            'Botnet' if self.label else 'Normal'
        ]


def write_aggregated_file(filename, directory):
    features = []
    basename = filename.split('.')[0].split('/')[1]
    with open(filename, 'r+') as f:
        with open(
                directory + '/'+basename + '.featureset.csv', 'w+') as out:
            headers = f.readline().strip().split(',')
            last = None
            bound = None
            current = list(range(4))
            feats = [
                'n_udp',
                'n_tcp',
                'n_icmp',
                'average_packet',
                'average_ttl',
                'average_fragment_count',
                'entropy_ws',
                'entropy_hs',
                'entropy_pnd',
                'entropy_pns',
                'entropy_tcp_flag',
                'label'
            ]
            out.write(','.join(feats) + '\n')
            summary = Summary()
            for line in f:
                info = line.strip().split(',')
                features = dict(zip(headers, info))
                time = datetime.strptime(features['start_time'], PCAP_FORMAT)

                if last is None or time > bound:
                    last = time
                    bound = last + timedelta(seconds=.15)
                    current = summary.to_feature_array()
                    out.write(','.join([str(x) for x in current]))
                    summary = Summary()
                    # Write the features to out.
                elif last < time < bound:
                    # Fill the features in.
                    summary.add(features)
            print('finished', filename)


if __name__ == '__main__':
    data = os.listdir('pcap_feature_label')
    data = ['pcap_feature_label/' + x for x in data]
    directory = 'aggregated_pcap'
    # directory = 'new_features_aggregation'
    for d in data:
        write_aggregated_file(d, directory)
