from SELFRec import SELFRec
from util.conf import ModelConf

if __name__ == '__main__':
    model = 'KGAT'
    conf = ModelConf(f'./conf/{model}.yaml')
    rec = SELFRec(conf)
    rec.execute()
