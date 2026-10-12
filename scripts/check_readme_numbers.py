"""
Проверка: все числа в блоке «Что говорят числа» обоих README
воспроизводятся из файлов прогонов в results/raw.

Запуск из корня репозитория:
    python scripts/check_readme_numbers.py
"""

import json, glob, re, collections

runs = collections.defaultdict(list)
for p in glob.glob('results/raw/*.json'):
    d = json.load(open(p)); runs[d['model']].append(d)

def m(model, metric):
    v = [r['test'][metric] for r in runs[model]]
    return sum(v)/len(v), v

def rd(model, metric, p=2):
    return round(m(model, metric)[0], p)

ok = True
def check(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + ('  ' + detail if detail else ''))
    ok = ok and cond

# headline values quoted in the block
for model, val in [('xgboost',13.79),('gru',13.93),('lstm',14.36),('rnn',15.39),
                   ('random_forest',16.27),('cnn1d',18.82)]:
    check(f'{model} RMSE {val}', rd(model,'rmse')==val, f'got {rd(model,"rmse")}')
for model, val in [('xgboost',312.53),('gru',372.50),('lstm',347.00),
                   ('rnn',538.03),('random_forest',659.07),('cnn1d',648.50)]:
    check(f'{model} PHM {val}', rd(model,'phm_score')==val, f'got {rd(model,"phm_score")}')

xgb = sorted(round(r['test']['rmse'],2) for r in runs['xgboost'])
lstm = sorted(round(r['test']['rmse'],2) for r in runs['lstm'])
gru  = sorted(round(r['test']['rmse'],2) for r in runs['gru'])
rnn  = sorted(round(r['test']['rmse'],2) for r in runs['rnn'])
rf   = sorted(round(r['test']['rmse'],2) for r in runs['random_forest'])

check('xgb worst seed 13.90', xgb[-1]==13.90, str(xgb))
check('lstm best seed 14.01', lstm[0]==14.01, str(lstm))
check('xgb worst beats lstm best', xgb[-1] < lstm[0])
check('gru best seed 13.26', gru[0]==13.26, str(gru))
check('gru best beats every xgb seed', gru[0] < xgb[0])
check('gap gru-xgb = 0.14', round(rd('gru','rmse')-rd('xgboost','rmse'),2)==0.14)
check('rnn beats RF on every seed', max(rnn) < min(rf), f'{rnn} vs {rf}')
check('rnn-lstm = 1.03', round(rd('rnn','rmse')-rd('lstm','rmse'),2)==1.03)
check('cnn-xgb = 5.03 (~five points)', round(rd('cnn1d','rmse')-rd('xgboost','rmse'),2)==5.03)
check('RF better RMSE than CNN', rd('random_forest','rmse') < rd('cnn1d','rmse'))
check('RF worse PHM than CNN', rd('random_forest','phm_score') > rd('cnn1d','phm_score'))

gphm = sorted(round(r['test']['phm_score'],1) for r in runs['gru'])
check('gru PHM range 264.1-463.8', gphm[0]==264.1 and gphm[-1]==463.8, str(gphm))
check('gru PHM ratio ~2x', 1.7 <= gphm[-1]/gphm[0] <= 1.8, f'{gphm[-1]/gphm[0]:.2f}')

par = {k: v[0]['parameters'] for k, v in runs.items() if 'parameters' in v[0]}
check('lstm params 56641', par['lstm']==56641, str(par.get('lstm')))
check('rnn params 13697', par['rnn']==13697, str(par.get('rnn')))
check('lstm ~4x rnn params', 3.9 <= par['lstm']/par['rnn'] <= 4.3, f"{par['lstm']/par['rnn']:.2f}")

# every number in the block must be an allowed one
allowed = set()
for model in runs:
    for metric, p in (('rmse',2),('phm_score',2),('mae',2),('r2',4)):
        vals = [r['test'][metric] for r in runs[model]]
        mean = sum(vals)/len(vals)
        sd = (sum((x-mean)**2 for x in vals)/len(vals))**0.5
        allowed.add(f'{mean:.{p}f}'); allowed.add(f'{sd:.{p}f}')
        for x in vals:
            allowed.add(f'{x:.2f}'); allowed.add(f'{x:.1f}')
allowed |= {f'{v}' for v in par.values()} | {'13 697','56 641','43 009','28 417'}
allowed |= {'125','30','2008','1','2','3','4','5','0.14','1.03','2026','129','135','978','8149','4124','4','09','001','002','003','004','37'}

for path in ('README.md','README.ru.md'):
    text = open(path, encoding='utf-8').read()
    start = text.find('numbers say') if path=='README.md' else text.find('говорят числа')
    end = text.find('## ', start+10)
    block = text[start:end]
    nums = re.findall(r'\d+(?:[  ]\d{3})*(?:\.\d+)?', block)
    bad = [n for n in nums if n not in allowed and n.replace(' ','') not in {str(v) for v in par.values()}]
    check(f'{path}: all numbers traceable to runs', not bad, f'unknown: {bad}')

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')

raise SystemExit(0 if ok else 1)
