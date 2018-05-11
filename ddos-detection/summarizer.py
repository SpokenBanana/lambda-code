import numpy as np
import collections
from scipy.stats import entropy as entropy_vector


class Summarizer:
    # Used to reduce amount of memory when aggregating. 
    __slots__ = ['bot', 'attack', 'features', 'entropy_features', 'std_features',
                 'data', '_ips', '_packets', '_bytes', '_time', '_time_counter',
                 '_states', '_dstports', '_srcports', 'src_to_dst', 'is_attack',
                 '_duration', 'used']

    def __init__(self, bot=None, attack=None):
        self.bot = bot
        self.attack = attack
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
            'n_d_a_p_address',
            'n_d_b_p_address',
            'n_d_c_p_address',
            'n_d_na_p_address',
            # New features
            'std_packets',
            'std_bytes',
            'std_time',
            'std_srcbytes',

            'src_to_dst',

            'entropy_sports>1024',
            'entropy_sports<1024',
            'entropy_dports>1024',
            'entropy_dports<1024',
            'entropy_srcport',
            'entropy_dstport',

            'entropy_dstip',

            # SRC related features, avoid them.
            # 'normal_flow_count',
            # 'background_flow_count',
            'n_s_a_p_address',
            'n_s_b_p_address',
            'n_s_c_p_address',
            'n_s_na_p_address',
            'entropy_srcip',
            'entropy_src_a_ip',
            'entropy_src_b_ip',
            'entropy_src_c_ip',
            'entropy_src_na_ip',

            'entropy_dst_a_ip',
            'entropy_dst_b_ip',
            'entropy_dst_c_ip',
            'entropy_dst_na_ip',

            'entropy_bytes',
            'entropy_src_bytes',
            'entropy_time',
            'entropy_state',
            'entropy_packets'
        ]

        # Just an easier way to do this since its very repetitive.
        self.entropy_features = {
            'entropy_time': collections.Counter(),
            'entropy_state': collections.Counter(),
            'entropy_srcport': collections.Counter(),
            'entropy_dstport': collections.Counter(),
            'entropy_srcip': collections.Counter(),
            'entropy_dstip': collections.Counter(),
            'entropy_bytes': collections.Counter(),
            'entropy_packets': collections.Counter(),
            'entropy_sports>1024': collections.Counter(),
            'entropy_sports<1024': collections.Counter(),
            'entropy_dports>1024': collections.Counter(),
            'entropy_dports<1024': collections.Counter(),
            'entropy_src_bytes': collections.Counter(),
            'entropy_src_a_ip': collections.Counter(),
            'entropy_src_b_ip': collections.Counter(),
            'entropy_src_c_ip': collections.Counter(),
            'entropy_dst_a_ip': collections.Counter(),
            'entropy_dst_b_ip': collections.Counter(),
            'entropy_dst_c_ip': collections.Counter(),
            'entropy_dst_na_ip': collections.Counter(),
            'entropy_src_na_ip': collections.Counter()
        }

        self.std_features = dict.fromkeys([
            'std_time',
            'std_packets',
            'std_srcbytes',
            'std_bytes'
        ], [])

        self.data = dict(zip(self.features, [0] * len(self.features)))

        self._ips = []
        self._packets = []
        self._bytes = []
        self._time = []
        self._time_counter = collections.Counter()
        self._states = collections.Counter()
        self._dstports = collections.Counter()
        self._srcports = collections.Counter()

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

        # Add to feature so we can calculate distribution by standard deviation.
        self._ips.append(item['srcaddr'])
        self.std_features['std_time'].append(float(item['dur']))
        self.std_features['std_packets'].append(float(item['totpkts']))
        self.std_features['std_bytes'].append(float(item['totbytes']))
        self.std_features['std_srcbytes'].append(float(item['srcbytes']))
        self._time_counter[item['dur']] += 1

        # Update all counters.
        self.entropy_features['entropy_time'][item['dur']] += 1
        self.entropy_features['entropy_state'][item['state']] += 1
        self.entropy_features['entropy_srcport'][item['sport']] += 1
        self.entropy_features['entropy_dstport'][item['dport']] += 1
        self.entropy_features['entropy_srcip'][item['srcaddr']] += 1
        self.entropy_features['entropy_dstip'][item['dstaddr']] += 1
        self.entropy_features['entropy_bytes'][item['totbytes']] += 1
        self.entropy_features['entropy_src_bytes'][item['srcbytes']] += 1
        self.entropy_features['entropy_packets'][item['totpkts']] += 1
        self._states[item['state']] += 1

        try:
            self._srcports[item['sport']] += 1
            self._dstports[item['dport']] += 1
        except Exception:
            pass  # Some port values are in a weird, non integer format, skip them.


        # Sometimes ports are in a weird format so exclude them for now
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

        src_class = classify(item['srcaddr'])
        dst_class = classify(item['dstaddr'])
        self.entropy_features['entropy_src_{}_ip'.format(src_class)][item['srcaddr']] += 1
        self.entropy_features['entropy_dst_{}_ip'.format(dst_class)][item['dstaddr']] += 1
        self.data['n_s_%s_p_address' % src_class] += 1
        self.data['n_d_%s_p_address' % dst_class] += 1

    def get_feature_list(self):
        """Returns all the feautres along with label as one list of strings."""
        for feat in self.std_features:
            self.data[feat] = np.std(self.std_features[feat])

        for feat in self.entropy_features:
            self.data[feat] = entropy(self.entropy_features[feat])

        self.data['src_to_dst'] = self.calc_src_to_dst()

        feature_list = []
        for key in self.features:
            feature_list.append(str(self.data[key]))

        feature_list.append('Botnet' if self.is_attack else 'Normal')
        if self.bot is not None or self.attack is not None:
            feature_list.append(str(self.bot) if self.is_attack else 'Normal')
            feature_list.append(str(self.attack))

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

    if 1 <= first <= 126:
        return 'a'
    elif 128 <= first <= 191:
        return 'b'
    elif 192 <= first <= 223:
        return 'c'
    return 'na'


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def standardize(x, mean, std):
    return (x - mean) / std