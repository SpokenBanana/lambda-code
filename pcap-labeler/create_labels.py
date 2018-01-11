"""Finds matches betweeen pcap packets and netflows.

Given a netflow file, a series of time ranges are created which pcap packets
will be matched up with by time and ip address.

PCAP FORMAT:
time,ipaddress

OUTPUT FILE:
time,ipaddress,label

The output file will then be used to label the pcap packets but this time
also write the features of the pcap packets along with the label.
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


NET_TIME = "%Y-%m-%d %H:%M:%S.%f"
TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"
PCAP_FORMAT = "%Y-%m-%d %H:%M:%S"

wstrip = re.compile('\s+')

def inet_to_str(inet):
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


class TimeRange:
    def __init__(self, start, dur, label, ip):
        self.start = start
        if dur <= 0.001:
            dur = 0.001
        self.end = start + timedelta(seconds=dur)
        self.label = label
        self.ip = ip

    def is_in_range(self, time, ip):
        return self.start <= time <= self.end and ip == self.ip

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'Start at: {}, end at: {}, and ip={}.'.format(
                self.start.strftime('%Y-%m-%d %H:%M:%S'),
                self.end.strftime('%Y-%m-%d %H:%M:%S'),
                self.ip)


def save_netflows(data):
    print('writing...')
    with open('netflow_ranges.pk1', 'wb') as f:
        pickle.dump(data, f)
    print('done')


def load_netflow_ranges():
    with open('netflow_ranges.pk1', 'rb') as f:
        time_ranges = pickle.load(f)
    return time_ranges


def load_binetflow_ranges():
    with open('time_ranges.pk1', 'rb') as f:
        time_ranges = pickle.load(f)
    return time_ranges


def get_time_ranges():
    """Gets the labeled time ranges for all 13 netflow files.

    Check first if a saved binetflow data is saved, if not, create it.
    """
    netflows = os.listdir('netflows')
    with Pool(10) as pool:
        results = pool.map(create_time_ranges, netflows)
    print('done')
    return results


def create_time_ranges(netflow):
    """Create time ranges from the netflow data.

    Must contain the start time, duration, source ip address, and label.
    """
    time_range = []
    ips = set()
    print('starting', netflow)
    with open('netflows/' + netflow, 'r+') as f:
        f.readline()
        for line in f:
            info = wstrip.split(line.strip())

            # Double check this with the netlfows.
            start_time = datetime.strptime(
                    info[0] + ' ' + info[1], NET_TIME)
            dur = float(info[2])
            label = info[-1]
            ip = info[4].split(':')[0]

            ips.add(ip)
            time_range.append(TimeRange(start_time, dur, label, ip))
    print('done with', netflow)
    return time_range, ips


def get_time_info(pcap_filename):
    base_name = pcap_filename.split('.')[0]
    with open('edited_pcaps/' + pcap_filename, 'rb') as f:
        with open('{}_info.txt'.format(base_name), 'w+') as out:
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


def match_time_ranges(time_range, pcap_filename):
    """Given time ranges, match up the packets in the pcap file to the ranges.

    Each range corresponds to a netflow that is labeled, so they should
    match up.

    Log process.

    Save labels with features in csv format.
    """
    total = 0
    labeled = 0
    for i in range(10):
        print(time_range[i])
    base_name = pcap_filename.split('.')[0]
    with open('edited_pcaps/' + pcap_filename, 'rb') as f:
        with open('labeled/' + base_name + '.csv', 'w+') as out:
            out.write('time,ip,label\n')
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                total += 1
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
                for times in time_range:
                    if times.is_in_range(time, str_ip):
                        out.write('{},{},{}\n'.format(
                            strtime, str_ip, times.label))
                        labeled += 1
                        break
                else:
                    print('None found for {} {}'.format(strtime, str_ip))
    print('{}/{} labeled'.format(labeled, total))

def separate_bots(pcap_filename):
    base_name = pcap_filename.split('.')[0]
    with open(base_name + '_info.txt', 'r+') as f:
        with open(base_name + '_botnets.txt', 'w+') as out:
            for line in f:
                _, ip = line.strip().split(',')
                if ip == '147.32.84.165':
                    out.write(line)


def separate_bots_from_netflow(pcap_name):
    base_name = pcap_name.split('.')[0]

    with open('netflows/' + base_name + '.pcap.netflow.labeled', 'r+') as f:
        with open(base_name + '.botnets.netflows.labeled', 'w+') as out:
            f.readline()
            for line in f:
                info = wstrip.split(line.strip())
                # Double check this with the netlfows.
                start_time = datetime.strptime(
                        info[0] + ' ' + info[1], NET_TIME)
                dur = float(info[2])
                label = info[-1]
                ip = info[4].split(':')[0]

                if ip == '147.32.84.165' or info[-1].strip() == 'Botnet':
                    out.write(line)


def match_time_with(time_range, pcap_filename, ips):
    total = 0
    labeled = 0

    start_time = datetime.strptime('2011-08-11 10:10:00.003', NET_TIME)

    base_name = pcap_filename.split('.')[0]
    with open('pcap_txt/' + base_name + '_info.txt', 'r+') as f:
        with open('labeled/'+base_name + '.matched.csv', 'w+') as out:
            print('opening')
            out.write('time,ip,label\n')
            pos = 0
            for line in f:
                total += 1
                str_time, str_ip = line.strip().split(',')
                str_time = str_time[:-3]
                time = datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
                # if time < start_time:
                #     continue

                if str_ip not in ips:
                    continue
                gone_over = 100
                over = 0
                last = pos
                while pos < len(time_range):
                    times = time_range[pos]
                    if times.is_in_range(time, str_ip):
                        out.write('{},{},{}\n'.format(str_time, str_ip,
                            times.label))
                        labeled += 1
                        break
                    if times.start > time:
                        over += 1
                        if over > gone_over:
                            print('gone over for {} | {}, labeled so far: {}'.format(
                                str_time, str_ip, labeled))
                            pos = last
                            break
                    pos += 1
                else:
                    print('not found {} | {}, labeled so far: {}'.format(
                        str_time, str_ip, labeled))
    print('{}/{}'.format(labeled, total))


def create_labels_for(pcap_files):
    """List of pcap files to label."""
    #time_ranges, ips = create_time_ranges(pcap_files[0].split('.')[0] + '.pcap.netflow.labeled')
    # time_ranges = [time_ranges]
    # time_ranges = [load_binetflow_ranges()[1]]
    for filename in pcap_files:
        print(filename)
        time_range, ips = create_time_ranges(filename)
        match_time_with(time_range, filename, ips)

if __name__ == '__main__':
    # files_to_label = ['capture20110811.truncated.pcap',
    #         'capture20110815-2.truncated.pcap']
    files_to_label = os.listdir('netflows')
    # get_time_info(files_to_label[0])
    create_labels_for(files_to_label)
    # separate_bots_from_netflow(files_to_label[1])
    # separate_bots(files_to_label[1])
    # save_netflows(get_time_ranges())
