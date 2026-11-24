#!/usr/bin/env python3
"""G01 QXU6028 Regular data: one-command L1/L2/L3 reproducible pipeline.
Integrated 24 November 2026 from the supplied L1/L2 checkpoint source and archived L3 checkpoint algorithm.
Run: python reproduce.py   (from this directory, or using its path)
"""
from __future__ import annotations
import os
# Use one BLAS thread for reproducibility and efficient small linear-algebra operations.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import argparse, hashlib, json, math, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CHECKS = []
FIGURES = ("L1_correction_factor.png", "model_figure.png", "trajectory_figure.png", "Figure_4_L3_map.png", "Figure_5_L3_validation.png")

def check(name, condition, detail=""):
    if not bool(condition):
        raise ValueError(f"{name}: {detail}")
    CHECKS.append(name)
    print(f"PASS: {name} {detail}", flush=True)

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False)+"\n", encoding="utf-8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_inputs(data):
    manifest=json.loads((data/"manifest.json").read_text())
    check("Manifest identity G01/regular", (manifest["group"],manifest["dataset_set"]) == ("G01","regular"))
    expected={"L1_data.csv","L1_params.json","L2_data.csv","L2_meta.json","L3_A.npy","L3_d.csv","L3_meta.json","README_FIRST.txt","results_template.json"}
    check("Manifest lists all nine source files", {x["name"] for x in manifest["files"]} == expected)
    for item in manifest["files"]:
        p=data/item["name"]
        check("SHA256 "+p.name, p.stat().st_size==item["bytes"] and sha(p)==item["sha256"])
    for name in ("L1_params.json","L2_meta.json","L3_meta.json","results_template.json"):
        obj=json.loads((data/name).read_text())
        check(name+" identity", (obj["group"],obj["dataset_set"]) == ("G01","regular"))

def version():
    # Use the actual integrated Git commit on a clone. A source-hash checked
    # release record supports running a ZIP extracted without Git installed.
    try:
        top=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()).resolve()
    except (OSError,subprocess.CalledProcessError):
        top=None
    if top==ROOT:
        status=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip()
        check("Tracked Git inputs and source unchanged",not status,status)
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    else:
        release=json.loads((ROOT/"PIPELINE_VERSION.json").read_text())
        check("Released source SHA256", sha(ROOT/"reproduce.py")==release["source_sha256"])
        commit=release["pipeline_commit"]
    check("Commit identifier valid",re.fullmatch(r"[0-9a-f]{40}",commit) is not None)
    return commit

KB_EV_K = 8.617333e-5
NA_CM2 = 3.816e15
E_CHARGE_C = 1.602176634e-19
EMAX_EV = 2.5


def fermi(E_eV: np.ndarray, EF_eV: float, T_K: float) -> np.ndarray:
    """Fermi-Dirac occupation, with energy in eV."""
    x = np.clip((E_eV - EF_eV) / (KB_EV_K * T_K), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))


def n_band(EF_eV: float, T_K: float, c: float) -> float:
    """Band-edge carrier count: integrate from E=0 to 2.5 eV."""
    E = np.linspace(0.0, EMAX_EV, 200_001)
    return float(NA_CM2 * np.trapezoid(c * E * fermi(E, EF_eV, T_K), E))


def n_fermi(EF_eV: float, T_K: float, c: float) -> float:
    """Fermi-referenced carrier count: integrate from E=EF to 2.5 eV."""
    E = np.linspace(EF_eV, EMAX_EV, 200_001)
    return float(NA_CM2 * np.trapezoid(c * E * fermi(E, EF_eV, T_K), E))


def n_fermi_closed(EF_eV: float, T_K: float, c: float) -> float:
    """Closed-form check for the Fermi-referenced integral."""
    kT = KB_EV_K * T_K
    return float(
        NA_CM2 * c * kT
        * (EF_eV * math.log(2.0) + (math.pi**2 / 12.0) * kT)
    )


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """Linearly interpolate the first x position where y crosses level."""
    i = np.where(np.diff(np.sign(y - level)) != 0)[0][0]
    return float(np.interp(level, y[i : i + 2], x[i : i + 2]))


def run_l1(data: Path, out: Path):
    params = json.loads((data / "L1_params.json").read_text())
    raw = pd.read_csv(data / "L1_data.csv")
    check("L1 input columns", raw.columns.tolist() == params["columns"])
    check("L1 raw sweep finite and ordered", np.isfinite(raw.to_numpy()).all() and np.all(np.diff(raw.EF_eV)>0))
    check("L1 sheet resistance matches metadata", np.allclose(raw.rho_ohm_sq, params["rho_ohm_sq"], rtol=1e-10))

    c = float(params["c"])
    T = float(params["T"])
    EF_op = float(params["EF_op_eV"])
    rho = float(params["rho_ohm_sq"])

    calculated_sweep = np.array([n_band(x, T, c) for x in raw.EF_eV])
    check("L1 supplied band-edge sweep agrees with model", np.allclose(calculated_sweep, raw.n_band_cm2, rtol=1e-5))
    # Published L1 gates.
    gates = {
        ef: n_band(ef, T, c) / n_fermi(ef, T, c)
        for ef in (0.0, 0.073, 0.100, 0.200)
    }
    assert abs(gates[0.0] - 1.00) < 2e-5
    assert abs(gates[0.073] - 2.00) < 0.03
    assert abs(gates[0.100] - 2.60) < 0.04
    assert abs(gates[0.200] - 5.10) < 0.06

    n_f_op = n_fermi(EF_op, T, c)
    n_f_closed = n_fermi_closed(EF_op, T, c)
    assert abs(n_f_op / n_f_closed - 1.0) < 2e-7

    correction = n_band(EF_op, T, c) / n_f_op
    mobility = 1.0 / (E_CHARGE_C * n_f_op * rho)

    EF = np.linspace(0.0, 0.25, 501)
    C = np.array([n_band(x, T, c) / n_fermi(x, T, c) for x in EF])
    ef2 = crossing(EF, C, 2.0)
    ef4 = crossing(EF, C, 4.0)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 160})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(EF, C, color="#176B87", linewidth=2.4,
            label=r"$C(E_F)=n_{band}/n_F$")
    ax.axhline(2, color="#999999", linewidth=0.9, linestyle="--")
    ax.axhline(4, color="#999999", linewidth=0.9, linestyle="--")
    ax.scatter([EF_op], [correction], color="#C0392B", s=45, zorder=3,
               label="Exact operating point")
    ax.scatter([ef2, ef4], [2, 4], color="#E67E22", s=30, zorder=3)
    ax.annotate(
        f"Operating point\n$E_F$={EF_op:.4f} eV, C={correction:.3f}",
        (EF_op, correction), xytext=(38, -52), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#C0392B"},
    )
    ax.annotate(f"C=2 at {ef2:.3f} eV", (ef2, 2),
                xytext=(-52, 19), textcoords="offset points")
    ax.annotate(f"C=4 at {ef4:.3f} eV", (ef4, 4),
                xytext=(8, 12), textcoords="offset points")
    ax.set(
        xlabel=r"Fermi level $E_F$ (eV)",
        ylabel="Carrier-count correction factor C",
        title="L1 carrier-count correction at 300 K",
    )
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0.8, max(6.2, float(C.max()) * 1.04))
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out / "L1_correction_factor.png", dpi=300)
    plt.close(fig)

    results = {
        "group": params["group"],
        "temperature_K": T,
        "EF_op_eV": EF_op,
        "correction_factor": correction,
        "n_fermi_op_cm2": n_f_op,
        "mobility_corrected_cm2_Vs": mobility,
        "EF_at_C2_eV": ef2,
        "EF_at_C4_eV": ef4,
        "gate_values": {str(k): v for k, v in gates.items()},
    }
    check("L1 correction monotonic", np.all(np.diff(C)>0))
    return {k: results[k] for k in ("correction_factor", "n_fermi_op_cm2", "mobility_corrected_cm2_Vs")}, results




def run_l2(data: Path, out: Path):
    plt.rcParams.update({"font.size": 10, "axes.spines.top":False,"axes.spines.right":False})
    raw=pd.read_csv(data/'L2_data.csv'); meta=json.loads((data/'L2_meta.json').read_text())
    check('Columns match metadata',raw.columns.tolist()==meta['columns'])
    check('Finite raw data',np.isfinite(raw.to_numpy()).all())
    check('Unique index-repeat pairs',not raw.duplicated(['measurement_index','repeat']).any())
    check('Three repeats at every index',raw.groupby('measurement_index')['repeat'].apply(lambda s:set(s)=={1,2,3}).all())
    check('Positive longitudinal resistance',(raw.R_xx_raw>0).all())
    indices=raw.measurement_index.drop_duplicates().to_numpy()
    check('Measurement order is increasing',np.all(np.diff(indices)>0))
    pairs=np.stack([raw[raw.measurement_index==i].sort_values('repeat')[['R_H_raw','R_xx_raw']].to_numpy() for i in indices])
    means=pairs.mean(axis=1); sd=pairs.std(axis=1,ddof=1)
    cross=np.flatnonzero(means[:-1,0]*means[1:,0]<0)
    check('Mean Hall sign crossing exists',len(cross)>0,[[int(indices[j]),int(indices[j+1])] for j in cross])

    def norm(x): return x/np.array([np.max(np.abs(x[:,0])),np.max(x[:,1])])
    xy=norm(means)
    check('Normalisation preserves Hall signs',np.array_equal(np.sign(xy[:,0]),np.sign(means[:,0])))
    check('Separate maxima equal one',np.isclose(np.abs(xy[:,0]).max(),1) and np.isclose(xy[:,1].max(),1))
    kBT=8.617333e-5*meta['T_K']; NA=3.816e15
    E=np.linspace(-2.5,2.5,4000); EF=np.linspace(-8*kBT,9*kBT,400)
    f=1/(np.exp(np.clip((E[None,:]-EF[:,None])/kBT,-60,60))+1)
    gaps=np.r_[0,np.arange(.5,8.001,.25)]
    curves=[]; populations_library=[]
    for gap in gaps:
        # The positive common DoS prefactor c cancels in each normalised channel.
        g=np.maximum(np.abs(E)-gap*kBT/2,0)
        n=NA*np.trapezoid(np.where(E[None,:]>=EF[:,None],g*f,0),E,axis=1)
        h=NA*np.trapezoid(np.where(E[None,:]<=EF[:,None],g*(1-f),0),E,axis=1)
        check(f'Finite positive populations: Eg/kBT={gap:g}',np.isfinite(n).all() and np.isfinite(h).all() and (n>0).all() and (h>0).all())
        populations_library.append((n.copy(),h.copy()))
        curves.append(norm(np.column_stack([(h-n)/(h+n)**2,1/(h+n)])))
    curves=np.asarray(curves)
    check('Library shape',curves.shape==(32,400,2),list(curves.shape))
    # Independent scalar integration verifies vectorisation and the prescribed masks.
    for m,j in [(0,0),(0,199),(16,210),(31,399)]:
        g=np.maximum(np.abs(E)-gaps[m]*kBT/2,0)
        ff=1/(np.exp(np.clip((E-EF[j])/kBT,-60,60))+1)
        nn=NA*np.trapezoid(np.where(E>=EF[j],g*ff,0),E)
        pp=NA*np.trapezoid(np.where(E<=EF[j],g*(1-ff),0),E)
        # symmetry tested separately at +/- 3 kBT using same symmetric grid
        check(f'Scalar integration matches vectorised result ({m},{j})',np.allclose([nn,pp],[populations_library[m][0][j],populations_library[m][1][j]],rtol=1e-12))
    for gap in [0,4,8]:
        def populations(ef):
            g=np.maximum(np.abs(E)-gap*kBT/2,0);ff=1/(np.exp(np.clip((E-ef)/kBT,-60,60))+1)
            return np.array([np.trapezoid(np.where(E>=ef,g*ff,0),E),np.trapezoid(np.where(E<=ef,g*(1-ff),0),E)])
        check(f'Electron-hole symmetry Eg/kBT={gap}',np.allclose(populations(3*kBT),populations(-3*kBT)[::-1],rtol=1e-10,atol=1e-16))
        z=populations(0);check(f'Neutrality Eg/kBT={gap}',np.isclose(z[0],z[1],rtol=1e-10,atol=1e-16))

    def fit(points):
        d2=((points[None,:,None,:]-curves[:,None,:,:])**2).sum(axis=-1)
        nearest=d2.argmin(axis=2)
        rms=np.sqrt(d2.min(axis=2).mean(axis=1))
        best=1+np.argmin(rms[1:]); chosen=int(best) if rms[best]<.85*rms[0] and gaps[best]>=1 else 0
        return chosen,nearest[chosen],rms,int(best)
    chosen,j,rms,best=fit(xy);trajectory=EF[j]
    # Independently loop over data points for the chosen curve.
    j_loop=np.array([np.argmin(np.sum((curves[chosen]-v)**2,axis=1)) for v in xy])
    check('Nearest-point vectorisation matches pointwise fit',np.array_equal(j,j_loop))
    check('Trajectory length equals unique indices',len(trajectory)==len(indices))
    check('No EF-grid boundary assignments',not np.any((j==0)|(j==399)))
    print('STEPS 5-6 COMPLETE',json.dumps({'model':'cone' if chosen==0 else 'hard_gap','Eg_kT':float(gaps[chosen]),'r_cone':float(rms[0]),'r_gap':float(rms[best])}),flush=True)

    seed=6028; rng=np.random.default_rng(seed); B=200
    boot_g=[];boot_ef=[];boot_selected=[]
    for b in range(B):
        picked=pairs[np.arange(len(indices)),rng.integers(0,3,size=len(indices)),:]
        c,jb,_,_=fit(norm(picked))
        boot_g.append(gaps[c]);boot_ef.append(EF[jb]);boot_selected.append(c>0)
    boot_g=np.asarray(boot_g);boot_ef=np.asarray(boot_ef)
    gq=np.percentile(boot_g,[16,50,84]);lo,hi=np.percentile(boot_ef,[16,84],axis=0)
    check('Bootstrap complete and finite',boot_ef.shape==(200,len(indices)) and np.isfinite(boot_ef).all())
    check('Bootstrap intervals ordered',np.all(lo<=hi))
    summary={'group':meta['group'],'T_K':meta['T_K'],'kBT_eV':kBT,'model':'cone' if chosen==0 else 'hard_gap','Eg_kT':float(gaps[chosen]),'Eg_eV':float(gaps[chosen]*kBT),'bootstrap_Eg_kT':{'p16':float(gq[0]),'median':float(gq[1]),'p84':float(gq[2])},'hard_gap_selection_fraction':float(np.mean(boot_selected)),'r_cone':float(rms[0]),'r_gap':float(rms[best]),'best_gap_candidate_kT':float(gaps[best]),'relative_residual_improvement':float(1-rms[best]/rms[0]),'measurement_indices':indices.tolist(),'EF_trajectory_eV':trajectory.tolist(),'EF_p16_eV':lo.tolist(),'EF_p84_eV':hi.tolist(),'bootstrap_seed':seed,'bootstrap_replicates':B,'zero_crossing_brackets':[[int(indices[x]),int(indices[x+1])] for x in cross]}
    fig,ax=plt.subplots(figsize=(7.6,5.4),layout='constrained')
    ax.plot(curves[0,:,0],curves[0,:,1],'-',color='#007b87',lw=1.8,label=f'Cone: RMS {rms[0]:.4f}')
    ax.plot(curves[best,:,0],curves[best,:,1],color='gray',ls='--',lw=1.2,label=f'Best hard gap: {gaps[best]:g} kBT; RMS {rms[best]:.4f}')
    sc=ax.scatter(xy[:,0],xy[:,1],c=indices,cmap='viridis',s=26,edgecolors='white',linewidths=.35,zorder=3)
    fig.colorbar(sc,ax=ax,label='Measurement index');ax.set(xlabel='Normalised Hall channel',ylabel='Normalised sheet-resistance channel',title=f"{meta['group']} L2 model comparison | selected: {summary['model']}")
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.14));ax.grid(alpha=.2)
    fig.savefig(out/'model_figure.png',dpi=300);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    ax.fill_between(indices,lo,hi,color='#007b87',alpha=.22,label='Bootstrap 16th-84th percentiles')
    ax.plot(indices,trajectory,'o-',ms=3,lw=1.3,color='#005b68',label='Fit to repeat means')
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Measurement index',ylabel='Assigned Fermi level (eV)',title=f"{meta['group']} L2 Fermi-level trajectory")
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'trajectory_figure.png',dpi=300);plt.close(fig)
    return {"model":summary["model"],"Eg_kT":summary["Eg_kT"],"Eg_uncertainty_kT":float((gq[2]-gq[0])/2),"EF_trajectory_eV":trajectory.tolist()}, summary


def gradient_matrix(n):
    rows=[]
    for i in range(n):
        for j in range(n):
            k=i*n+j
            for h in ([k+1] if j<n-1 else [])+([k+n] if i<n-1 else []):
                row=np.zeros(n*n); row[k]=-1; row[h]=1; rows.append(row)
    return np.array(rows)


def run_l3(data: Path, out: Path, threshold: float):
    # Archived September L3 checkpoint: projected first-difference inverse,
    # Cholesky factorisation, 1e-9 numerical stabiliser, and local extrema.
    meta=json.loads((data/'L3_meta.json').read_text()); n=int(meta['npix']); N=n*n
    A=np.load(data/'L3_A.npy',allow_pickle=False); d=np.loadtxt(data/'L3_d.csv',delimiter=',')
    check('L3 dimensions', n==10 and A.shape==(128,N) and d.shape==(3,128))
    check('L3 finite readings',np.isfinite(A).all() and np.isfinite(d).all())
    row_sum=float(np.max(np.abs(A.sum(axis=1))))
    check('L3 uniform offset is unobservable',row_sum<1e-11,row_sum)
    P=np.eye(N)-np.ones((N,N))/N; L=gradient_matrix(n)
    eta0=.01*(np.linalg.norm(A,2)/np.linalg.norm(L,2))**2
    def solver(eta):
        M=P@(A.T@A+eta*L.T@L)@P+1e-9*np.eye(N)
        fact=cho_factor(M)
        def solve(y):
            x=cho_solve(fact,P@(A.T@y)); return x-x.mean(axis=0)
        return solve
    dbar=d.mean(axis=0); solve=solver(eta0); m=solve(dbar).reshape(n,n)
    check('L3 map finite and zero mean',np.isfinite(m).all() and abs(m.mean())<1e-12)
    sd=d.std(axis=0,ddof=1); rng=np.random.default_rng(20260924)
    noise=rng.normal(size=(128,500))*sd[:,None]
    noise_peaks=np.max(np.abs(solve(noise)),axis=0)
    delta=float(np.quantile(noise_peaks,.95)); coords=np.linspace(-2.7,2.7,n)
    features=[]
    for sign in (1,-1):
        z=sign*m; candidates=[]
        for i in range(n):
            for j in range(n):
                block=z[max(i-1,0):min(i+2,n),max(j-1,0):min(j+2,n)]
                if z[i,j]>delta and z[i,j]>=block.max(): candidates.append((z[i,j],i,j))
        for v,i,j in sorted(candidates,reverse=True):
            if all(max(abs(i-f['row']),abs(j-f['column']))>1 for f in features):
                features.append(dict(sign='positive' if sign==1 else 'negative',x_probe_pitch=float(coords[j]),y_probe_pitch=float(coords[i]),amplitude=float(m[i,j]),row=i,column=j))
    widths=[]; psfs=[]; yy,xx=np.meshgrid(coords,coords,indexing='ij')
    for i,j in ((5,5),(2,7)):
        q=np.zeros(N); q[n*i+j]=1; q-=q.mean()
        point=solve(A@q).reshape(n,n); w=np.maximum(point,0)
        sigma2=np.sum(w*((xx-coords[j])**2+(yy-coords[i])**2))/(2*w.sum())
        widths.append(float(2.355*np.sqrt(sigma2))); psfs.append(point)
    sens=[]
    for mult in (.01,.1,1,10,100):
        x=solver(eta0*mult)(dbar)
        sens.append(dict(eta=float(eta0*mult),multiplier=mult,residual_norm=float(np.linalg.norm(A@x-dbar)),roughness_norm=float(np.linalg.norm(L@x)),peak=float(np.max(abs(x)))))
    peak=float(np.max(np.abs(m))); verdict=peak<=threshold
    if peak+delta<threshold: confidence=f'Clear pass: peak + 95% noise-only detection limit = {peak+delta:.6g} < threshold {threshold:.6g}.'
    elif peak-delta>threshold: confidence=f'Clear fail: peak - 95% noise-only detection limit = {peak-delta:.6g} > threshold {threshold:.6g}.'
    else: confidence=f'Borderline: nominal {"pass" if verdict else "fail"}; threshold {threshold:.6g} overlaps the noise envelope.'
    confidence+=' Repeat noise only; systematic reconstruction bias is excluded.'
    # Figure 4: row 0 at the lower edge, consistent with physical coordinates.
    fig,ax=plt.subplots(figsize=(7.2,6.0),layout='constrained')
    im=ax.imshow(m*100,origin='lower',extent=(-3,3,-3,3),cmap='RdBu_r',vmin=-peak*100,vmax=peak*100,interpolation='nearest')
    fig.colorbar(im,ax=ax,label='Relative conductivity deviation (%)')
    ax.set(xlabel='x (probe pitches)',ylabel='y (probe pitches)',title='G01 L3 reconstructed zero-mean conductivity map')
    for f in features:
        x,y=f['x_probe_pitch'],f['y_probe_pitch']
        ax.scatter([x],[y],s=140,facecolors='none',edgecolors='black',linewidths=1.5,zorder=3)
        dx=-14 if x>0 else 12; dy=-19 if y>0 else 14
        ax.annotate(f"{f['amplitude']*100:+.3f}%",(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if x>0 else 'left',fontsize=10,bbox=dict(fc='white',ec='none',alpha=.88,pad=2),arrowprops=dict(arrowstyle='-',lw=.8))
    fig.savefig(out/'Figure_4_L3_map.png',dpi=300); plt.close(fig)
    # Figure 5 combines all validation evidence into one six-panel report figure.
    fig,axs=plt.subplots(2,3,figsize=(12,7.4),layout='constrained')
    ax=axs[0,0]; sv=np.linalg.svd(A,compute_uv=False)
    ax.semilogy(np.arange(1,len(sv)+1),np.maximum(sv,np.finfo(float).tiny),lw=1.5); ax.set(xlabel='Singular-value index',ylabel='Singular value',title='(a) Singular-value spectrum'); ax.grid(alpha=.2)
    ax=axs[0,1]; res=[s['residual_norm'] for s in sens]; rough=[s['roughness_norm'] for s in sens]
    ax.loglog(res,rough,'o-',lw=1.5); ax.scatter(res[2],rough[2],s=100,facecolors='none',edgecolors='black',linewidths=1.5)
    for s in sens: ax.annotate(f"{s['multiplier']:g}η₀",(s['residual_norm'],s['roughness_norm']),xytext=(3,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel=r'Residual norm $\|Am-\bar d\|_2$',ylabel=r'Roughness $\|Lm\|_2$',title='(b) Regularisation trade-off'); ax.grid(alpha=.2)
    ax=axs[0,2]; ax.semilogx([s['multiplier'] for s in sens],[100*s['peak'] for s in sens],'o-',lw=1.5,label='Reconstructed peak'); ax.axhline(100*threshold,ls='--',color='gray',label='3% limit' if threshold==.03 else 'Limit'); ax.scatter([1],[100*peak],s=100,facecolors='none',edgecolors='black'); ax.set(xlabel='Weight multiplier η/η₀',ylabel='Maximum absolute deviation (%)',title='(c) Peak sensitivity'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    v=max(np.abs(p).max() for p in psfs)
    for k,(ax,point,rc,width) in enumerate(zip(axs[1,:2],psfs,((5,5),(2,7)),widths)):
        i,j=rc; im=ax.imshow(point,origin='lower',extent=(-3,3,-3,3),cmap='RdBu_r',vmin=-v,vmax=v,interpolation='nearest'); ax.plot(coords[j],coords[i],marker='+',color='black',ms=10,mew=1.5); ax.set(xlabel='x (probe pitches)',ylabel='y (probe pitches)',title=f'({"de"[k]}) Impulse {rc}\nFWHM = {width:.3f} pitches')
    fig.colorbar(im,ax=list(axs[1,:2]),label='Reconstructed impulse amplitude',shrink=.78)
    ax=axs[1,2]; ax.hist(100*noise_peaks,bins=22,edgecolor='white'); ax.axvline(delta*100,ls='--',lw=1.7,color='black',label=f'95% = {delta*100:.5f}%'); ax.set(xlabel='Noise-only maximum deviation (%)',ylabel='Count',title='(f) Noise detection limit (500 trials)'); ax.legend(fontsize=8)
    fig.savefig(out/'Figure_5_L3_validation.png',dpi=300); plt.close(fig)
    result=dict(map=m.ravel().tolist(),features=features,resolution_probe_pitch=max(widths),detection_limit_fraction=delta,**{'pass':bool(verdict)},confidence=confidence)
    diag=dict(eta0=float(eta0),row_sum_max=row_sum,map_mean=float(m.mean()),peak_fraction=peak,noise_sd_median=float(np.median(sd)),noise_sd_mean=float(sd.mean()),noise_trials=500,noise_seed=20260924,FWHM_tests_probe_pitches=widths,regularisation_sensitivity=sens,singular_values=sv.tolist(),threshold_fraction=threshold,threshold_source='3% documented in archived G01 regular L3 checkpoint README; verify against actual course announcement',noise_envelope=[peak-delta,peak+delta])
    return result,diag


def validate(result,template,threshold):
    check('Exact top-level and phase keys match supplied template',set(result)==set(template) and all(set(result[k])==set(template[k]) for k in ('L1','L2','L3')))
    def finite(v):
        if v is None: return False
        if isinstance(v,dict): return all(finite(x) for x in v.values())
        if isinstance(v,list): return all(finite(x) for x in v)
        if isinstance(v,(float,int)): return math.isfinite(v)
        return isinstance(v,str)
    check('No null or non-finite values',finite(result))
    check('Results identity',result['schema_version']=='QXU6028-results-v1' and result['group']=='G01' and result['dataset_set']=='regular')
    check('L1 positive numeric outputs',all(type(v) in (float,int) and v>0 for v in result['L1'].values()))
    l2=result['L2']; l3=result['L3']; m=np.asarray(l3['map'])
    check('L2 exactly 56 finite energy assignments',len(l2['EF_trajectory_eV'])==56 and np.isfinite(l2['EF_trajectory_eV']).all())
    check('L2 model and nonnegative gap uncertainty',l2['model'] in ('cone','hard_gap') and l2['Eg_kT']>=0 and l2['Eg_uncertainty_kT']>=0)
    check('L3 exactly 100 finite zero-mean pixels',m.shape==(100,) and np.isfinite(m).all() and abs(m.mean())<1e-12)
    check('L3 positive resolution and detection limit',l3['resolution_probe_pitch']>0 and l3['detection_limit_fraction']>0)
    check('L3 feature coordinates and amplitude match map',len(l3['features'])>=2 and all(set(f)=={'sign','row','column','x_probe_pitch','y_probe_pitch','amplitude'} and 0<=f['row']<10 and 0<=f['column']<10 and f['amplitude']==m[10*f['row']+f['column']] and np.isclose(f['x_probe_pitch'],-2.7+.6*f['column']) and np.isclose(f['y_probe_pitch'],-2.7+.6*f['row']) and f['sign']==('positive' if f['amplitude']>0 else 'negative') for f in l3['features']))
    check('L3 boolean verdict matches stated threshold',type(l3['pass']) is bool and l3['pass']==bool(np.abs(m).max()<=threshold))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--output',type=Path,default=ROOT/'output')
    parser.add_argument('--threshold',type=float,default=.03)
    args=parser.parse_args(); data=args.data; out=args.output
    check('Threshold finite and positive',math.isfinite(args.threshold) and args.threshold>0)
    verify_inputs(data); commit=version(); out.mkdir(parents=True,exist_ok=True)
    l1,d1=run_l1(data,out); l2,d2=run_l2(data,out); l3,d3=run_l3(data,out,args.threshold)
    template=json.loads((data/'results_template.json').read_text()); result=dict(template)
    result.update(pipeline_commit=commit,L1=l1,L2=l2,L3=l3); validate(result,template,args.threshold)
    write_json(out/'results.json',result); check('JSON round-trip equality',json.loads((out/'results.json').read_text())==result)
    check('All five report figures generated',all((out/n).stat().st_size>0 for n in FIGURES))
    write_json(out/'diagnostics.json',{'L1':d1,'L2':d2,'L3':d3,'pipeline_commit':commit,'versions':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}})
    write_json(out/'validation_checks.json',{'all_passed':True,'checks':CHECKS,'official_schema_file':'data/results_template.json','output_sha256':{n:sha(out/n) for n in ('results.json',*FIGURES)}})
    print('Completed G01/regular; L1 + L2 + L3 and five figures in',out)
    print('L1:',l1,'L2:',{k:v for k,v in l2.items() if k!='EF_trajectory_eV'},'L3 peak:',d3['peak_fraction'],'L3 pass:',l3['pass'])

if __name__=='__main__': main()
