"""Writes the pcap files into a txt with their time and ip address to label.

To match pcap packets to the netflows, we just need the time and ip address.
It takes some time to parse through each pcap packet so having them in a
file format makes it easier to read through as well as faster to process.
"""
from datetime import datetime, timedelta
from multiprocessing import Pool
import multiprocessing
import os
import dpkt
import numpy
import pickle
import socket
import re
from create_labels import inet_to_str

def get_time_info(pcap_filename):
    base_name = pcap_filename.split('.')[0]
    with open('edited_pcaps/' + pcap_filename, 'rb') as f:
        with open('pcap_txt/{}_info.txt'.format(base_name), 'w+') as out:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                time = datetime.fromtimestamp(ts)
                time += timedelta(hours=6)
                strtime = time.strftime('%Y-%m-%d %H:%M:%S.%f')

                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    str_ip = inet_to_str(ip.src)
                except AttributeError as e:
                    print(e)
                    continue
                out.write('{},{}\n'.format(strtime, str_ip))


if __name__ == '__main__':
    pcaps = os.listdir('edited_pcaps')
    # pcap = 'capture20110815-2.truncated.pcap'
    with Pool(10) as pool:
        results = pool.map(get_time_info, pcaps)
        print('done with all')
