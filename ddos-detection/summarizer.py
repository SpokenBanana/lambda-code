import numpy as np
import collections
from scipy.stats import entropy as entropy_vector


class Summarizer:
    def __init__(self):
        self.features = [
            'n_conn',
            'avg_duration',
            'n_udp',
            'n_tcp',
            'n_icmp',
            'n_sports>1024',
            'n_sports<1024',
            'n_dports>1024',
            'n_dports<1024',
            'n_s_a_p_address',
            'n_s_b_p_address',
            'n_s_c_p_address',
            'n_d_a_p_address',
            'n_d_b_p_address',
            'n_d_c_p_address',
            'n_d_na_p_address',
            'n_s_na_p_address',
            'normal_flow_count',
            'background_flow_count',
            # New features
            'entropy_srcip',
            'entropy_dstip',
            'std_packet',
            'entropy_srcport',
            'entropy_dstport',
            'std_bytes',
            'std_time',
            'entropy_time',
            'entropy_state',
            # TODO: Investigate how to add the new interesting feature.
            'src_to_dst',
            # TODO: Add std and entropy of new features. Also as entropy time.
            'entropy_sports>1024',
            'entropy_sports<1024',
            'entropy_dports>1024',
            'entropy_dports<1024'
        ]

        # Just an easier way to do this since its very repetitive.
        self.entropy_features = {
            'entropy_time': collections.Counter(),
            'entropy_state': collections.Counter(),
            'entropy_srcport': collections.Counter(),
            'entropy_dstport': collections.Counter(),
            'entropy_srcip': collections.Counter(),
            'entropy_dstip': collections.Counter(),
            'entropy_sports>1024': collections.Counter(),
            'entropy_sports<1024': collections.Counter(),
            'entropy_dports>1024': collections.Counter(),
            'entropy_dports<1024': collections.Counter()
        }

        self.data = dict(zip(self.features, [0] * len(self.features)))

        self._ips = []
        self._packets = []
        self._dstports = collections.Counter()
        self._srcports = collections.Counter()
        self._bytes = []
        self._time = []
        self._time_counter = collections.Counter()
        self._states = collections.Counter()

        self.src_to_dst = {}

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

        if item['srcaddr'] not in self.src_to_dst:
            dstcounter = collections.Counter()
            dstcounter[item['dstaddr']] += 1
            byte_counter = collections.Counter()
            byte_counter[item['totbytes']] += 1
            time_counter = collections.Counter()
            time_counter[item['dur']] += 1
            self.src_to_dst[item['srcaddr']] = [dstcounter,
                    byte_counter, time_counter]
        else:
            sofar = self.src_to_dst[item['srcaddr']]
            sofar[0][item['dstaddr']] += 1
            sofar[1][item['totbytes']] += 1
            sofar[2][item['dur']] += 1

        self._ips.append(item['srcaddr'])
        self._time.append(float(item['dur']))
        self._packets.append(float(item['totpkts']))
        self._bytes.append(float(item['totbytes']))
        self._time_counter[item['dur']] += 1

        self.entropy_features['entropy_time'][item['dur']] += 1
        self.entropy_features['entropy_state'][item['state']] += 1
        self.entropy_features['entropy_srcport'][item['sport']] += 1
        self.entropy_features['entropy_dstport'][item['dport']] += 1
        self.entropy_features['entropy_srcip'][item['srcaddr']] += 1
        self.entropy_features['entropy_dstip'][item['dstaddr']] += 1

        try:
            self._srcports[item['sport']] += 1
            self._dstports[item['dport']] += 1
        except Exception:
            pass
        self._states[item['state']] += 1

        # sometimes ports are in a weird format so exclude them for now
        try:
            if int(item['sport']) < 1024:
                self.data['n_sports<1024'] += 1
                self.entropy_features['entropy_sports<1024'][item['sport']] += 1
            else:
                self.data['n_sports>1024'] += 1
                self.entropy_features['entropy_sports>1024'][item['sport']] += 1
        except Exception:
            pass

        try:
            if int(item['dport']) < 1024:
                self.data['n_dports<1024'] += 1
                self.entropy_features['entropy_dports<1024'][item['dport']] += 1
            else:
                self.data['n_dports>1024'] += 1
                self.entropy_features['entropy_dports>1024'][item['dport']] += 1
        except Exception:
            pass

        if 'Botnet' in item['label']:
            self.is_attack = 1
        elif 'Normal' in item['label']:
            self.data['normal_flow_count'] += 1
        elif 'Background' in item['label']:
            self.data['background_flow_count'] += 1

        # TODO: Add entropy for each of these cases.
        # entropy_features['{}_ip'.format(_class)][item['addr']] += 1
        src_class = classify(item['srcaddr'])
        dst_class = classify(item['dstaddr'])
        self.data['n_s_%s_p_address' % src_class] += 1
        self.data['n_d_%s_p_address' % dst_class] += 1

    def get_feature_list(self):
        """Returns all the feautres along with label as one list of strings."""
        self.data['std_packet'] = np.std(self._packets)
        self.data['std_time'] = np.std(self._time)
        self.data['std_bytes'] = np.std(self._bytes)

        for feat in self.entropy_features:
            self.data[feat] = entropy(self.entropy_features[feat])

        self.data['src_to_dst'] = self.calc_src_to_dst()

        feature_list = []
        for key in self.features:
            feature_list.append(str(self.data[key]))

        feature_list.append('Botnet' if self.is_attack else 'Normal')
        return feature_list

    def calc_src_to_dst(self):
        values = list(self.src_to_dst.values())
        values = [(entropy(x[0]), entropy(x[1]), entropy(x[2])) for x in values]
        entropy_values = [entropy_vector(x) for x in values]
        final_entropy = entropy_vector(entropy_values)
        if str(final_entropy) == '-inf':
            return 0
        return final_entropy


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
