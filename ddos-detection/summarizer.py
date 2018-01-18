import numpy as np
import collections
from utils import get_feature_order

class Summarizer:
    def __init__(self):
        self.data = {
            'n_conn': 0,
            'avg_duration': 0,
            'n_udp': 0,
            'n_tcp': 0,
            'n_icmp': 0,
            'n_sports>1024': 0,
            'n_sports<1024': 0,
            'n_dports>1024': 0,
            'n_dports<1024': 0,
            'n_s_a_p_address': 0,
            'n_s_b_p_address': 0,
            'n_s_c_p_address': 0,
            'n_s_na_p_address': 0,
            'n_d_a_p_address': 0,
            'n_d_b_p_address': 0,
            'n_d_c_p_address': 0,
            'n_d_na_p_address': 0,
            'normal_flow_count': 0,
            'background_flow_count': 0,
            # New features
            'std_ip_a': 0,
            'std_ip_b': 0,
            'std_ip_c': 0,
            'std_packet': 0,
            'std_srcport': 0,
            'std_dstport': 0,
            'std_bytes': 0,
            'std_time': 0,
            'std_state': 0
            # TODO: Investigate how to add the new interesting feature.
        }

        self._ips = []
        self._packets = []
        self._dstports = []
        self._srcports = []
        self._bytes = []
        self._time = []
        self._states = []

        self.is_attack = 0  # would be 1 if it is an attack, set 0 by default
        self._duration = 0
        self.used = False

    def add(self, item):
        self.used = True
        self.data['n_conn'] += 1

        proto = 'n_%s' % item['proto']
        if proto in self.data:
            self.data[proto] += 1

        self._duration += float(item['dur'])
        self.data['avg_duration'] = self._duration / self.data['n_conn']

        self._ips.append(item['srcaddr'])
        self._time.append(float(item['dur']))
        self._packets.append(float(item['totpkts']))
        self._bytes.append(float(item['totbytes']))
        try:
            self._srcports.append(int(item['dport']) if item['dport'] else 0)
            self._dstports.append(int(item['sport']) if item['sport'] else 0)
        except Exception:
            pass
        # TODO: Add states.

        # sometimes ports are in a weird format so exclude them for now
        try:
            if int(item['sport']) < 1024:
                self.data['n_sports<1024'] += 1
            else:
                self.data['n_sports>1024'] += 1
        except Exception:
            pass

        try:
            if int(item['dport']) < 1024:
                self.data['n_dports<1024'] += 1
            else:
                self.data['n_dports>1024'] += 1
        except Exception:
            pass

        if 'Botnet' in item['label']:
            self.is_attack = 1
        elif 'Normal' in item['label']:
            self.data['normal_flow_count'] += 1
        elif 'Background' in item['label']:
            self.data['background_flow_count'] += 1

        self.data['n_s_%s_p_address' % classify(item['srcaddr'])] += 1
        self.data['n_d_%s_p_address' % classify(item['dstaddr'])] += 1

    def get_feature_list(self):
        """Returns all the feautres along with label as one list of strings."""
        self.data['entropy_ip_a'] = entropy(self._ips)
        self.data['std_packet'] = np.std(self._packets)
        self.data['std_time'] = np.std(self._time)
        self.data['std_bytes'] = np.std(self._bytes)
        self.data['std_srcport'] = np.std(self._srcports)
        self.data['std_dsrtport'] = np.std(self._dstports)

        feature_list = []
        for key in get_feature_order():
            feature_list.append(str(self.data[key]))

        feature_list.append('Botnet' if self.is_attack else 'Normal')
        return feature_list


def entropy(items):
    C = collections.Counter(items)
    counts = np.array(list(C.values()), dtype=float)
    prob = counts / counts.sum()
    return (-prob * np.log2(prob)).sum()


def classify(ip):
    parts = ip.split('.')
    try:
        first = int(parts[0])
    except Exception:
        return 'na'

    # TODO: write a better way to classify this.
    if 1 <= first <= 126:
        return 'a'
    elif 128 <= first <= 191:
        return 'b'
    elif 192 <= first <= 223:
        return 'c'
    return 'na'


if __name__ == '__main__':
    s = Summarizer()
    print(list(s.data.keys()))
