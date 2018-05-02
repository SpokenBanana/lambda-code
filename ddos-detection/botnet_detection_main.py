from absl import app
from absl import flags

from botnet_detection import summary_of_detection

FLAGS = flags.FLAGS

# TODO: Merge these into a single flag.
flags.DEFINE_bool(
    'use_bots', False, 'Whether or not to use bots as the label.')
flags.DEFINE_bool(
    'use_attacks', False, 'Whether or not to use attack as the label')
flags.DEFINE_bool(
    'sample', False, 'Whether or not to sample from Normal labels.')
flags.DEFINE_bool(
    'use_ahead', False, 'Whether or not to use attack as the label')

flags.DEFINE_bool(
    'use_background', False, 'Use the file that has background information.')
flags.DEFINE_string('attack_type', None, 'Type of attack to train on.')
flags.DEFINE_string('model_type', None, 'Type of model to train with.')
flags.DEFINE_float('interval', None, 'Interval of the file to train on.')
flags.DEFINE_bool(
    'norm_and_standardize', False, 'To normalize and standardize the features')


def main(_):
    if FLAGS.use_ahead:
        base_name = 'minute_aggregated/capture2011081{}-{}_ahead.aggregated.csv'
        f = base_name.format(FLAGS.attack_type, FLAGS.interval)
    else:
        base_name = 'minute_aggregated/{}{}-{}s.featureset.csv'
        f = base_name.format(
            FLAGS.attack_type,
            '' if not FLAGS.use_background else '_background',
            FLAGS.interval)

    result = summary_of_detection(
        f, FLAGS.model_type, FLAGS.use_bots, FLAGS.use_attacks,
        FLAGS.use_attack, FLAGS.sample, use_ahead=FLAGS.use_ahead,
        norm_and_standardize=FLAGS.norm_and_standardize)
    print(result)
    print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}".format(
        *result))


if __name__ == '__main__':
    app.run(main)
