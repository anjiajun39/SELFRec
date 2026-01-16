from SELFRec import SELFRec
from util.conf import ModelConf
import time
import os

if __name__ == '__main__':
    model = 'KGRec'
    s = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(base_dir, 'conf', f'{model}.yaml')
    conf = ModelConf(conf_path)
    conf.config['max.epoch'] = 2
    conf.config['batch.size'] = 512
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print(f"Running time: {e - s:.2f} s")
