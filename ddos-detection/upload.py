import dropbox
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('path', None, 'Path to file to upload.')

with open('token.txt', 'r+') as f:
    TOKEN = f.read().strip()

dbx = dropbox.Dropbox(TOKEN)

def upload_file(filename):
    name = filename.split('/')[-1]
    with open(filename, 'rb') as f:
        data = f.read()
    try:
        dbx.files_upload(
            data, '/{}'.format(name), dropbox.files.WriteMode.overwrite
        )
    except dropbox.exceptions.ApiError as err:
        print(err)
    print('DONE')


def main(_):
    upload_file(FLAGS.path)


if __name__ == '__main__':
    app.run(main)
