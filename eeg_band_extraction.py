"""Reprocesado completo desde los datos brutos (solo lectura) para theta, alfa y beta en las 18 condiciones del Excel.
- Clasificación de ensayos por marcadores: códigos de diana S4/S5 = High (Go/No-Go), S6/S7 = Medium, S8/S9 = Low;
  resultado: S1 = correcto, S2 = falsa alarma (No-Go), S3 = Go fallido -> tardío (con respuesta) u omisión (sin respuesta).
  Condiciones: Go_X (correcto), Go_X_AllResp (correcto + tardío), Go_X_Late, Go_X_Omission, NoGo_X (correcto), NoGo_X_FA.
- Preprocesado: 0.75-40 Hz; canales ruidosos (SD por época) interpolados; referencia promedio; ICA (picard, 15) con Fp1/Fp2
  como proxy EOG; épocas -3.0..+2.3 s respecto al marcador; rechazo ±90 µV (sin Fp1/Fp2); CSD (Legendre 10).
- Potencia: Morlet 4-25 Hz, 1 Hz, 3->10 ciclos; ROI theta Fz/FC1/FC2 (4-7 Hz), alfa P7/P8/P3/P4/O1/O2 (8-12 Hz),
  beta CP5/CP1/C3 (15-25 Hz). Potencia absoluta por condición (promedio de ensayos).
- Realineado: diana real = pico del ERP occipital (O1/Oz/O2) en 0.5-0.8 s tras el marcador - 0.09 s (mediana si pico < 3 µV).
Salida: bandas_realineado_raw.npz (sesión x condición x banda x tiempo, potencia absoluta, tiempo del marcador) + meta."""
import os, sys, re, numpy as np, pandas as pd, mne
mne.set_log_level('ERROR')
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = [os.environ.get("EEG_RAW_DIR", os.path.join(HERE, "data", "raw"))]   # BrainVision files (.vhdr/.vmrk/.eeg; not included)
DIF = {4: 'High', 5: 'High', 6: 'Medium', 7: 'Medium', 8: 'Low', 9: 'Low'}
CONDS = [f'{p}_{d}{s}' for d in ('Low', 'Medium', 'High') for p, s in
         [('Go', ''), ('Go', '_AllResp'), ('Go', '_Late'), ('Go', '_Omission'), ('NoGo', ''), ('NoGo', '_FA')]]
BANDS = {'Theta': ((4, 7), ['Fz', 'FC1', 'FC2']), 'Alpha': ((8, 12), ['P7', 'P8', 'P3', 'P4', 'O1', 'O2']),
         'Beta': ((15, 25), ['CP5', 'CP1', 'C3'])}
FREQS = np.arange(4., 26.); NCYC = 3 + (FREQS - 4) * 7 / 21
PICKS = sorted({c for _, (_, ch) in BANDS.items() for c in ch})
OCC = ['O1', 'Oz', 'O2']

def find(sub, ext='.vhdr'):
    for r in RAW:
        p = os.path.join(r, sub + ext)
        if os.path.exists(p): return p

def trial_categories(sub):
    mk = sorted([(m.group(1)[0], m.group(2).strip(), int(m.group(3))) for l in open(find(sub, '.vmrk'), errors='ignore')
                 for m in [re.match(r'Mk\d+=(Stimulus|Response),([^,]*),(\d+)', l)] if m], key=lambda x: x[2])
    tg = [i for i, x in enumerate(mk) if x[0] == 'S' and x[1] in {f'S  {k}' for k in DIF}]
    out = []
    for a, i in enumerate(tg):
        s = mk[i][2]; end = mk[tg[a + 1]][2] if a + 1 < len(tg) else s + 2000
        seg = [x for x in mk[i + 1:] if x[2] < end]
        res = next((x[1] for x in seg if x[0] == 'S'), None); resp = any(x[0] == 'R' for x in seg)
        k = int(mk[i][1].split()[-1]); d = DIF[k]
        if k in (4, 6, 8):
            cat = f'Go_{d}' if res == 'S  1' else (f'Go_{d}_Late' if resp else f'Go_{d}_Omission')
        else:
            cat = f'NoGo_{d}' if res == 'S  1' else f'NoGo_{d}_FA'
        out.append((s, cat))
    return out

def process(sub):
    raw = mne.io.read_raw_brainvision(find(sub), preload=True)
    raw.pick([c for c in raw.ch_names if c not in ('ECG', 'EOG')])
    raw.set_montage('easycap-M1', on_missing='ignore')
    raw.filter(0.75, 40., picks='eeg')
    tc = trial_categories(sub)
    ev = np.array([[s, 0, i + 1] for i, (s, _) in enumerate(tc)])
    cats = np.array([c for _, c in tc])
    ep = mne.Epochs(raw, ev, tmin=-3.0, tmax=2.3, baseline=None, preload=True)
    cats = cats[ep.selection]
    X = ep.get_data(picks='eeg'); chsd = np.median(X.std(axis=2), axis=0); names = np.array(ep.copy().pick('eeg').ch_names)
    ep.info['bads'] = list(names[(chsd > 3 * np.median(chsd)) | (chsd < 0.1 * np.median(chsd))]); nbad = len(ep.info['bads'])
    if nbad: ep.interpolate_bads(reset_bads=True)
    ep.set_eeg_reference('average')
    good_fit = np.abs(ep.get_data(picks='eeg')).max(axis=(1, 2)) < 500e-6
    ica = mne.preprocessing.ICA(n_components=15, method='picard', random_state=97, max_iter=500)
    ica.fit(ep[good_fit].copy().filter(1., None), decim=3)
    idx = []
    for ch in ('Fp1', 'Fp2'):
        i, _ = ica.find_bads_eog(ep[good_fit], ch_name=ch, threshold=3.0); idx += i
    ica.exclude = sorted(set(idx))[:3]; ica.apply(ep)
    rej_ch = [c for c in ep.copy().pick('eeg').ch_names if c not in ('Fp1', 'Fp2')]
    ok = np.abs(ep.get_data(picks=rej_ch)).max(axis=(1, 2)) < 90e-6
    ep, cats = ep[ok], cats[ok]
    erp = ep.copy().pick(OCC).average().data.mean(0)
    te = ep.times
    m = (te >= 0.5) & (te <= 0.8); pk_t, pk_a = te[m][np.argmax(erp[m])], erp[m].max() * 1e6
    ep = mne.preprocessing.compute_current_source_density(ep, n_legendre_terms=10)
    groups = {c: np.where(cats == c)[0] for c in CONDS}
    groups.update({f'Go_{d}_AllResp': np.where(np.isin(cats, [f'Go_{d}', f'Go_{d}_Late']))[0] for d in ('Low', 'Medium', 'High')})
    curves, ntr, t_tf = None, {}, None
    for j, c in enumerate(CONDS):
        ii = groups[c]; ntr[c] = len(ii)
        if len(ii) == 0: continue
        pw = ep[ii].compute_tfr('morlet', freqs=FREQS, n_cycles=NCYC, picks=PICKS, average=True, decim=5, return_itc=False)
        if curves is None:
            t_tf = pw.times; curves = np.full((len(CONDS), len(BANDS), len(t_tf)), np.nan)
        for b, (bn, ((lo, hi), ch)) in enumerate(BANDS.items()):
            ci = [pw.ch_names.index(x) for x in ch]; fi = (pw.freqs >= lo) & (pw.freqs <= hi)
            curves[j, b] = pw.data[np.ix_(ci, fi)].mean(axis=(0, 1))
    return t_tf, curves, ntr, te, erp, pk_t, pk_a, int(ok.sum()), len(ok), nbad

if __name__ == '__main__':
    meta = pd.read_csv(os.path.join(HERE, 'tf_bands_meta.csv'))
    subs = meta if len(sys.argv) == 1 else meta[meta.Subject.isin(sys.argv[1:])]
    rows, C, E, T = [], [], [], None
    for _, r in subs.iterrows():
        try:
            t_tf, curves, ntr, te, erp, pk_t, pk_a, n_ok, n_all, nb = process(r.Subject)
        except Exception as e:
            print('ERROR', r.Subject, e, flush=True); continue
        T = t_tf; C.append(curves); E.append(erp)
        rows.append(dict(Subject=r.Subject, BaseSubject=r.BaseSubject, Fase=r.Fase, n_clean=n_ok, n_all=n_all, n_interp=nb,
                         erp_peak_t=pk_t, erp_peak_uV=pk_a, **{f'n_{c}': ntr[c] for c in CONDS}))
        print(r.Subject, r.Fase, n_ok, '/', n_all, 'pico ERP %.3f s %.1f µV' % (pk_t, pk_a), flush=True)
    if len(sys.argv) == 1:
        np.savez(os.path.join(HERE, 'bandas_realineado_raw.npz'), P=np.array(C), t=T, erp=np.array(E), t_erp=te,
                 conds=np.array(CONDS), bands=np.array(list(BANDS)))
        pd.DataFrame(rows).to_csv(os.path.join(HERE, 'bandas_realineado_meta.csv'), index=False)
