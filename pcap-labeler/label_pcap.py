"""Take the output file from the label_pcap.py to get a labeled pcap file.

In each file of the matched pcap packets, if there is a pcaket that
matches the time and ip address, then write all the features of the
packet along with the label in another file

INPUT FILE: (.featureset.csv)
time,ipaddress,label

OUTPUT FILE:
:features,label

We now have each pcap file labeled. They just need to be aggregated with
aggregator.py
"""
from datetime import datetime, timedelta
from multiprocessing import Pool
import multiprocessing
import os
from os.path import basename
import dpkt
import numpy
import pickle
import socket
import re
from create_labels import inet_to_str

PCAP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

def get_files(directory):
    files = os.listdir(directory)
    return ['{}/{}'.format(directory, name) for name in files]

class Label:
    def __init__(self, time, ip):
        self.time = time
        self.ip = ip
        # self.rep = ' '.join([time, ip])

    def __hash__(self):
        return hash((self.time, self.ip))

    def __eq__(self, other):
        return (self.time, self.ip) == (other.time, other.ip)

    def __ne__(self, other):
        return not (self == other)


def find_label_for(pcap, info):
    base = basename(pcap)
    time_and_ip = {}
    with open(info, 'r+') as f:
        for line in f:
            time, ip, label = line.strip().split(',')
            lab = Label(time, ip)
            time_and_ip[lab] = label

    labeled = 0
    with open(pcap, 'rb') as fpcap:
        with open('pcap_feature_label/{}.featureset.csv'.format(base),
                'w+') as out:
            header = [
                'start_time',
                'packet_length',
                'fragment_offset',
                'TTL',
                'proto',
                'label'
            ]
            out.write(','.join(header) + '\n')
            pcaps = dpkt.pcap.Reader(fpcap)
            print('starting')
            for ts, buf in pcaps:
                time = datetime.fromtimestamp(ts)
                time += timedelta(hours=6)
                strtime = time.strftime(PCAP_FORMAT)
                strtime = strtime[:-3]

                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    str_ip = inet_to_str(ip.src)
                except AttributeError as e:
                    continue

                key = Label(strtime, str_ip)

                if key in time_and_ip:
                    b = bytearray()
                    b.extend(buf)
                    b = b[14:]
                    blen = len(b)
                    if blen < 20:
                        continue

                    features = [
                        strtime,
                        str((b[2] << 8) + b[3]),  # packet length
                        str(int(b[7] > 0)),  # fragment offset
                        str(b[8]),  # TTL
                        str(b[9]),  # protocol of IP payload
                        time_and_ip[key]
                        # TODO: Extend this featureset once more is known.
                    ]
                    out.write(','.join(features)+ '\n')
                    labeled += 1
    print('{}/{}'.format(labeled, len(time_and_ip)))


if __name__ == '__main__':
    pcap_files = sorted(get_files('edited_pcaps'))
    pcap_files.remove('edited_pcaps/capture20110818-2.truncated.pcap')
    info_files = sorted(get_files('labeled'))

    for pcap, info in zip(pcap_files, info_files):
        print('starting', pcap, info)
        find_label_for(pcap, info)


