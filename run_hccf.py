from SELFRec import SELFRec
from util.conf import ModelConf
import time

if __name__ == '__main__':
    model = 'HCCF'
    s = time.time()
    conf = ModelConf(f'./conf/{model}.yaml')
    # Set small epoch for quick verification
    conf.config['max.epoch'] = 5
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print(f"Running time: {e - s:.2f} s")
