import json
import datetime

d = dict()
d['Created'] = datetime.datetime.now().strftime('%d%b%Y')
d['owner'] = 'Guy Barash'
d['version'] = 'Bulbasaur'
d['bank_files'] = 3
d['glados'] = r'https://bitbucket.wdc.com/projects/SD-CE-INAND/repos/dst-womb/browse/STM'
d['what'] = r'https://www.youtube.com/watch?v=B9RIHOnGGsg'

with open(r"C:\work\dst-womb\STM\dump\Glados\glados_metadata.json", 'w') as fp:
    json.dump(d, fp, indent=4)
