from SELFRec import SELFRec
from util.conf import ModelConf
import time

if __name__ == '__main__':
    model = 'DHCF'
    s = time.time()
    conf = ModelConf(f'./conf/{model}.yaml')
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print(f"Running time: {e - s:.2f} s")
