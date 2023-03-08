import argparse
import os
import io
import sys
import pickle
import random

HIGH_LANGS = 'ar,bg,bn,bs,ca,cs,da,de,el,es,et,eu,fa,fi,fr,he,hr,hu,id,is,it,ja,ko,lt,lv,mk,ms,mt,nl,no,pl,pt,ro,ru,si,sk,sl,sq,sr,sv,th,tr,uk,vi,zh'.split(
    ',')
MED_LANGS = 'af,as,az,br,cy,eo,ga,gl,gu,hi,ka,km,ku,mg,ml,nb,ne,nn,pa,rw,sh,ta,tg,tt,ur,uz,wa,xh'.split(',')
LOW_LANGS = 'am,be,fy,gd,ha,ig,kk,kn,ky,li,mr,my,oc,or,ps,se,te,tk,ug,yi,zu'.split(',')
N = 94


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str, default=r'/path/to/GTrans/model/BLEU/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert len(HIGH_LANGS) + len(MED_LANGS) + len(LOW_LANGS) == N
    x2x = {}
    for lg in HIGH_LANGS + MED_LANGS + LOW_LANGS:
        with open(f"{args.log}/en-{lg}.BLEU", "r", encoding="utf-8") as r:
            lines = r.readlines()
            fill = False
            for i in range(len(lines) - 1, 0):
                if 'BLEU+case.mixed+lang.' in lines[i]:
                    x2x[f'en-{lg}'] = float(lines[i].split()[2])
                    fill = True
            if not fill:
                print(f"Error en-{lg}")
        with open(f"{args.log}/{lg}-en.BLEU", "r", encoding="utf-8") as r:
            lines = r.readlines()
            fill = False
            for i in range(len(lines) - 1, 0):
                if 'BLEU+case.mixed+lang.' in lines[i]:
                    x2x[f'{lg}-en'] = float(lines[i].split()[2])
                    fill = True
            if not fill:
                print(f"Error {lg}-en")

    assert len(x2x) == N * 2
    e2x = list(filter(lambda x: x > 0, [x2x[key] if 'en-' in key else -1 for key in x2x.keys()]))
    e2x_high = list(
        filter(lambda x: x > 0, [x2x[key] if key.split('-')[1] in HIGH_LANGS else -1 for key in x2x.keys()]))
    e2x_med = list(filter(lambda x: x > 0, [x2x[key] if key.split('-')[1] in MED_LANGS else -1 for key in x2x.keys()]))
    e2x_low = list(filter(lambda x: x > 0, [x2x[key] if key.split('-')[1] in LOW_LANGS else -1 for key in x2x.keys()]))
    x2e = list(filter(lambda x: x > 0, [x2x[key] if '-en' in key else -1 for key in x2x.keys()]))
    x2e_high = list(
        filter(lambda x: x > 0, [x2x[key] if key.split('-')[0] in HIGH_LANGS else -1 for key in x2x.keys()]))
    x2e_med = list(filter(lambda x: x > 0, [x2x[key] if key.split('-')[0] in MED_LANGS else -1 for key in x2x.keys()]))
    x2e_low = list(filter(lambda x: x > 0, [x2x[key] if key.split('-')[0] in LOW_LANGS else -1 for key in x2x.keys()]))
    assert len(e2x) == N and len(x2e) == N
    print("e2x | High: {:.2f} Med: {:.2f} Low: {:.2f} Avg: {:.2f}".format(sum(e2x_high) / len(e2x_high),
                                                                          sum(e2x_med) / len(e2x_med),
                                                                          sum(e2x_low) / len(e2x_low),
                                                                          sum(e2x) / len(e2x)))
    print("x2e | High: {:.2f} Med: {:.2f} Low: {:.2f} Avg: {:.2f}".format(sum(x2e_high) / len(x2e_high),
                                                                          sum(x2e_med) / len(x2e_med),
                                                                          sum(x2e_low) / len(x2e_low),
                                                                          sum(x2e) / len(x2e)))
    LINE_WIDTH = 14
    print("en->x result")
    for i in range(0, N, LINE_WIDTH):
        print("& {} \\\\".format(" &".join([str(s) for s in e2x[i: i + LINE_WIDTH]]).strip()))

    print("x->en result")
    for i in range(0, N, LINE_WIDTH):
        print("& {} \\\\".format(" &".join([str(s) for s in x2e[i: i + LINE_WIDTH]]).strip()))
