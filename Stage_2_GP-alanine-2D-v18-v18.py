#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-file alanine GAPS with the supplied working GAPS-US execution pattern.

Run ONCE in Spyder on CRC bastion, using spyder_env. The v18-v17 controller
runs in this foreground Spyder console. Only MD windows are submitted to SGE.
Controller progress and per-step plots appear directly in Spyder.

50x50 periodic grid; initial 50-point right-edge psi line; five queries per AL step;
query+bridge nodes submitted together as an SGE array; up to 50 AL iterations
and 1000 paid nodes. Fixed phi/psi constraints, no CV springs. Original GP
fitting, acquisition and accept/reject logic retained. Existing accepted MD
samples are reused within this campaign; active files use compute-visible group storage.
No archive results are imported: this is a clean, fixed-cost MD benchmark.
MD logs are compressed on node-local disk before shared-disk copy-back.
Stopping Spyder pauses the controller but leaves submitted MD workers running.
Rerun to resume. The worker must be invoked only with --cp2k-worker by its job.
The numerical self-test can be run with --single-selftest.
"""
import os
from pathlib import Path
import sys

# ----------------------------- USER SETTINGS --------------------------------
SINGLE_PROJECT = os.environ.get('GAPS_STAGED_SOURCE', '/users/mfarshad/DeltaG/GAPS_v18_v17_ALANINE_APPENDED')
SINGLE_CODE_REVISION = 'fixed-cost-spyder-one-replica-v1'
SUPERSEDED_CONTROLLER_JOBS = ('1754380','1752864','1751309')
_SPYDER_RUN_ID = None
_SPYDER_CONTROLLER_LOCK = None
_SPYDER_PLOT_HISTORY = {}
_SPYDER_PLOT_VIEWER = {}
SINGLE_CAMPAIGN = 'ALANINE_SPYDER_V18_V17_RIGHT_EDGE_B5_FIXED_COST_1REP'
SINGLE_STORAGE_ROOT = '/groups/jwhitme1/mfarshad/GAPS_ALANINE_GROUP'  # Compute-visible storage; keep home free.
GOLD_ARCHIVE_ROOT = '/goldfs/jwhitme1/Data-MF-01/GAPS_ALANINE_ACTIVE'
GRID_RESOLUTION = 50
MAX_PARALLEL_WINDOWS = 50  # SGE array cap; each window requests two CPU slots for one CP2K replica.
LIVE_REFRESH_SECONDS = 3
COMPATIBLE_IO_CODE_SHA256S = ('76a69925da7aaa45e99c4f3c0ad419e0361a78cac649b417a9b60a8ac309c665', 'b71cec19f037fc7fc499bd3ccc31bf8e94eb0572ef85fa57c0d2b945d58eb0a7', 'a5c4481778dab8d719b2fc8f78d82b0d638d29833d0c01a14669d01339f870e5', '9db70369755627b42ca927841343183178c414752f916373cd1d8a24ac60bdf7', '7c188337b7c8463145f0d99ee62fdcb3d7d7e1985ee1c7dd52c3b78f7c618934', '4aaeb032c0d464ad65a9553095dd89d03c65944df7223e893a3db2b9b13b0546', '348a7b6d699e6e77042277e32c8bddf027e9e0667e329316f49f490bcb89c92c')
COMPATIBLE_PREVIOUS_SINGLE_SHA256S = ()  # New initial-line policy uses its own campaign.
SCIENTIFIC_CODE_SHA256 = '4db2d10e0a462f2078abfabe9bae147d4deff2b99ab1baad49eaf88cc8d230e6'
# GP kernel, cold/warm start, AL batch and AL iterations remain in their ORIGINAL
# locations below (m32+rq*se, cold, batch 5, 50 iterations in the attached file).
# ---------------------------------------------------------------------------


def _single_normalize_ast(node):
    """Canonical syntax representation shared by Python 3.10 and 3.12.

    Ignore locations, comments, docstrings and 3.12's empty type_params field;
    retain actual parameters, expressions, algorithm and literal values.
    """
    import ast
    if isinstance(node,ast.AST):
        result={'node':type(node).__name__}
        for field,value in ast.iter_fields(node):
            if field=='type_params' or (isinstance(node,ast.Constant) and field=='kind'):continue
            if field=='body' and isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                if value and isinstance(value[0],ast.Expr) and isinstance(value[0].value,ast.Constant) and isinstance(value[0].value.value,str):value=value[1:]
            result[field]=_single_normalize_ast(value)
        return result
    if isinstance(node,list):return [_single_normalize_ast(value) for value in node]
    if isinstance(node,bytes):return {'bytes':node.hex()}
    if node is Ellipsis:return {'literal':'Ellipsis'}
    if isinstance(node,complex):return {'complex':[node.real,node.imag]}
    return node


def _single_scientific_signature(text):
    import ast,hashlib,json
    lines=text.splitlines(keepends=True)
    label='COMPLETE JOINT-CV GEOMETRY AND MEAN-FORCE CODE'
    markers=[i for i,line in enumerate(lines) if line.lstrip().startswith('#')
             and ' '.join(line.lstrip()[1:].split())==label]
    if len(markers)!=1:
        raise RuntimeError(f'Expected one scientific section heading; found {len(markers)}. Replace the whole file with the downloaded version; do not append it.')
    tree=ast.parse(''.join(lines[markers[0]+1:]))
    canonical=json.dumps(_single_normalize_ast(tree),sort_keys=True,separators=(',',':'),ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _single_backend_signature():
    return _single_scientific_signature(Path(__file__).read_text())






def _single_campaign_root():
    base=Path(SINGLE_STORAGE_ROOT)
    if not base.is_dir():raise RuntimeError('Active shared storage is unavailable: '+str(base))
    return base/SINGLE_CAMPAIGN

def _single_boot_status(stage,**extra):
    import json,socket,time
    project=Path(SINGLE_PROJECT);root=_single_campaign_root()
    root.mkdir(parents=True,exist_ok=True)
    record=dict(stage=stage,time=time.asctime(),run_id=_SPYDER_RUN_ID,
                host=socket.gethostname(),pid=os.getpid())
    record.update(extra)
    record['controller_location']='sge_compute_node' if os.environ.get('JOB_ID') else 'spyder_foreground'
    record['code_revision']=SINGLE_CODE_REVISION
    tmp=root/f'status.json.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,root/'status.json')
    return record


def _single_prepare_spyder(batch=False):
    """Follow the supplied GAPS-US pattern: controller here, MD jobs via qsub."""
    import fcntl,getpass,shutil,socket,subprocess,time,uuid
    global _SPYDER_RUN_ID,_SPYDER_CONTROLLER_LOCK
    if batch:
        if os.environ.get('JOB_ID') in (None,'','undefined'):
            raise RuntimeError('--run-batch must be launched through qsub')
        if os.environ.get('SGE_TASK_ID') not in (None,'','undefined'):
            raise RuntimeError('A controller must not run as an SGE array task')
    elif os.environ.get('JOB_ID') not in (None,'','undefined'):
        raise RuntimeError('Interactive controller requires bastion/Spyder; use --run-batch inside an SGE controller job')
    project=Path(SINGLE_PROJECT).resolve()
    if not project.is_dir():raise RuntimeError('Staged source project unavailable: '+str(project))
    if batch and not str(project).startswith(str(Path(SINGLE_STORAGE_ROOT).resolve())+'/'):
        raise RuntimeError('Compute controller requires staged project on compute-visible group storage')
    if _single_backend_signature()!=SCIENTIFIC_CODE_SHA256:
        raise RuntimeError('Parameters or algorithm differ from this packaged version. Comments/formatting are allowed; use the complete updated file for this run.')
    if GRID_RESOLUTION!=50:raise ValueError('This requested configuration uses a 50x50 grid')
    if not isinstance(MAX_PARALLEL_WINDOWS,int) or MAX_PARALLEL_WINDOWS<1:raise ValueError('MAX_PARALLEL_WINDOWS must be positive')
    for command in ('qsub','qstat','qdel'):
        if not shutil.which(command):raise RuntimeError(command+' is unavailable; start Spyder with the CRC environment')
    q=subprocess.run(['qstat','-u',getpass.getuser()],capture_output=True,text=True,check=True)
    old_jobs=[]
    for line in q.stdout.splitlines():
        f=line.split()
        if len(f)>3 and f[0].isdigit() and f[2].startswith(('GAPS_SINGL','GAPS_CP2K')):
            if f[0] not in SUPERSEDED_CONTROLLER_JOBS:raise RuntimeError('Another compute-node controller is active: '+f[0])
            old_jobs.append(f[0])
    for job in old_jobs:
        subprocess.run(['qdel',job],check=True)
        for _ in range(60):
            if subprocess.run(['qstat','-j',job],capture_output=True).returncode!=0:break
            time.sleep(1)
        else:raise RuntimeError('Old controller deletion is still pending; rerun after it leaves the queue')
    if any(len(line.split())>2 and line.split()[2].startswith('GAPS_ALA') for line in q.stdout.splitlines()):
        raise RuntimeError('Old GAPS MD workers are still queued/running. Stop them before changing campaign storage/initialization.')
    root=_single_campaign_root();root.mkdir(exist_ok=True)
    lock=(root/'adapter.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise RuntimeError('This campaign already has another active controller')
    _SPYDER_CONTROLLER_LOCK=lock;_SPYDER_RUN_ID=uuid.uuid4().hex
    os.chdir(project)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
        os.environ[name]='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    _single_boot_status('loading numerical libraries; no MD submitted yet')
    print('\nGAPS-ENERGY CONTROLLER ON '+('SGE COMPUTE NODE' if batch else 'SPYDER / BASTION'),flush=True)
    print('Code revision:',SINGLE_CODE_REVISION,flush=True)
    print('Host:',socket.gethostname(),'Python:',sys.executable,flush=True)
    print('Grid: 50x50; queries/AL step: 5; initial right-edge psi line: 50 points; maximum AL steps: 50.',flush=True)
    print('CP2K window arrays are separately submitted to SGE by this controller.',flush=True)
    print('Loading TensorFlow/GPflow; completed MD samples will be reused.',flush=True)
    try:
        from IPython import get_ipython
        ip=get_ipython()
        if ip is not None:ip.run_line_magic('matplotlib','inline')
        else:
            import matplotlib
            matplotlib.use('Agg')
    except Exception as exc:print('Plot backend configuration:',exc,flush=True)


def _single_close_controller():
    global _SPYDER_CONTROLLER_LOCK
    handle=_SPYDER_CONTROLLER_LOCK
    if handle is not None:
        try:handle.close()
        except Exception:pass
        _SPYDER_CONTROLLER_LOCK=None


def _single_startup_failed(exc):
    if _SPYDER_RUN_ID is not None:_single_boot_status('STOPPED during library import',error=str(exc))
    _single_close_controller()




def _single_manifest_compatible(previous,current):
    old=dict(previous);new=dict(current)
    a=old.pop('scientific_code_sha256',None);b=new.pop('scientific_code_sha256',None)
    # Only audited I/O, initialization and bounded sampling-policy repairs may
    # migrate unchanged base configuration, grid and original-controller identity.
    return old==new and (a==b or (a in COMPATIBLE_IO_CODE_SHA256S and b==SCIENTIFIC_CODE_SHA256))


def _single_plot_coordinates(points):
    """Display each periodic point once, using +pi for the identified seam."""
    import numpy as np
    result=np.asarray(points,float).copy()
    result[np.isclose(result,-np.pi,atol=1.e-10,rtol=0)]=np.pi
    return result


def _single_periodic_plot_segments(a,b):
    """Keep periodic crossings, but draw boundary-coincident edges only once."""
    import numpy as np
    a=np.asarray(a,float);b=np.asarray(b,float)
    end=a+(b-a+np.pi)%(2*np.pi)-np.pi
    for x in (-2*np.pi,0,2*np.pi):
        for y in (-2*np.pi,0,2*np.pi):
            shift=np.array([x,y]);first=a+shift;last=end+shift
            if np.any(np.maximum(first,last)<-np.pi-1.e-10) or np.any(np.minimum(first,last)>np.pi+1.e-10):continue
            # The negative and positive boundaries are identical. Use the
            # right/top copy for paths lying exactly on either boundary.
            if any(np.isclose(first[k],-np.pi,atol=1.e-10,rtol=0) and
                   np.isclose(last[k],-np.pi,atol=1.e-10,rtol=0) for k in (0,1)):continue
            yield first,last


def _single_plot_anchor(ax,point):
    p=_single_plot_coordinates(point)
    ax.scatter(p[0],p[1],marker='*',s=165,c='gold',edgecolors='black',
               linewidths=.8,zorder=20,clip_on=False,label='Reference anchor (F = 0)')


def _single_replot_saved():
    """Read completed analysis snapshots only; never import GPflow or submit MD."""
    import matplotlib
    matplotlib.use('Agg')
    flag=sys.argv.index('--replot')
    root=Path(sys.argv[flag+1]).resolve() if flag+1<len(sys.argv) else _single_campaign_root()
    destination=root/'plots_display_v12';history={};viewer={};count=0
    snapshots=sorted((root/'analysis').glob('step_*.npz'))
    if not snapshots:raise RuntimeError('No saved analysis snapshots found under '+str(root/'analysis'))
    for path in snapshots:
        try:
            if _single_display_analysis(path,history,viewer,save_directory=destination,show=False):count+=1
        except Exception as exc:
            print(f'[REPLOT SKIP] {path.name}: {exc}',flush=True)
    if not count:raise RuntimeError('No complete snapshots could be replotted; existing results were not changed')
    print(f'[REPLOT COMPLETE] {count} snapshots; corrected PNG/PDF figures: {destination}',flush=True)


def _single_display_analysis(path,history,viewer,save_directory=None,show=True):
    import io
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter,MaxNLocator
    metadata=path.with_suffix('.json')
    if not metadata.exists():return False
    with np.load(path,allow_pickle=False) as data:
        p=data['phi_psi_rad'];mean=data['mean_kbt'];sigma=data['sigma_kbt']
        reference=data['external_reference_kbt'];paid=data['paid_nodes'].astype(int)
        edges=data['mst_edges'].astype(int)
    info=json.loads(metadata.read_text())
    if int(info['paid_nodes'])!=len(paid):raise ValueError('Snapshot metadata/data are being updated; retry later')
    anchor=int(info.get('anchor_node',0))
    if anchor not in set(paid.tolist()):raise ValueError('Reference anchor is not a paid node')
    phi=np.unique(p[:,0]);psi=np.unique(p[:,1])
    if len(phi)*len(psi)!=len(p):raise ValueError('Analysis grid is not rectangular')
    ix=np.searchsorted(phi,p[:,0]);iy=np.searchsorted(psi,p[:,1])
    def grid(values):
        result=np.full((len(psi),len(phi)),np.nan);result[iy,ix]=values
        return result
    step=int(info['step']);history[step]=info
    backend=plt.get_backend().lower();inline=('inline' in backend or backend=='agg')
    fig=viewer.get('figure')
    if inline or fig is None or not plt.fignum_exists(fig.number):
        fig=plt.figure(figsize=(15,8),constrained_layout=False)
        if not inline:viewer['figure']=fig
    else:fig.clear()
    layout=fig.add_gridspec(2,6,left=.065,right=.94,bottom=.08,top=.88,wspace=1.1,hspace=.65)
    axes=[fig.add_subplot(layout[0,2*i:2*i+2]) for i in range(3)]
    for ax,values,title,cmap in zip(axes,(mean,sigma,np.abs(mean-reference)),
                                  ('GP free energy','GP uncertainty','External-FES diagnostic error'),
                                  ('viridis','viridis','magma')):
        z=grid(values);limits={}
        lo=float(np.nanmin(z));hi=float(np.nanmax(z))
        if hi-lo<1e-10*max(1.,abs(lo),abs(hi)):
            limits=dict(vmin=0.,vmax=max(1e-6,hi*1.1)) if title!='GP free energy' else dict(vmin=lo-.1,vmax=hi+.1)
        # Repeat the seam value to cover the whole periodic plotting domain.
        image=ax.pcolormesh(np.r_[phi,np.pi],np.r_[psi,np.pi],
                           np.pad(z,((0,1),(0,1)),mode='wrap'),shading='nearest',cmap=cmap,**limits)
        bar=fig.colorbar(image,ax=ax,fraction=.045,pad=.035)
        bar.set_label('kBT',fontsize=10);bar.ax.tick_params(labelsize=9)
        bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('phi (rad)',fontsize=10);ax.set_ylabel('psi (rad)',fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_xlim(-np.pi,np.pi);ax.set_ylim(-np.pi,np.pi);ax.set_aspect('equal')
    for a,b in edges.reshape(-1,2):
        for first,last in _single_periodic_plot_segments(p[a],p[b]):
            axes[0].plot([first[0],last[0]],[first[1],last[1]],color='lightgrey',lw=.7)
    displayed=_single_plot_coordinates(p[paid])
    axes[0].scatter(displayed[:,0],displayed[:,1],s=10,c='tab:orange',edgecolors='black',linewidths=.2,zorder=5,clip_on=False)
    _single_plot_anchor(axes[0],p[anchor])
    ordered=[history[k] for k in sorted(history)]
    rmse=fig.add_subplot(layout[1,:3]);counts=fig.add_subplot(layout[1,3:])
    rmse.plot([x['paid_nodes'] for x in ordered],[x['external_reference_RMSE_kbt'] for x in ordered],'-o')
    rmse.set_xlabel('Paid MD nodes',fontsize=10);rmse.set_ylabel('External-FES diagnostic RMSE (kBT)',fontsize=10);rmse.grid(alpha=.2)
    rmse.set_xlim(0,max(1,max(x['paid_nodes'] for x in ordered)*1.1))
    rmse.set_ylim(0,max(1e-6,max(x['external_reference_RMSE_kbt'] for x in ordered)*1.1))
    rmse.xaxis.set_major_locator(MaxNLocator(integer=True))
    counts.plot([int(x['step']) for x in ordered],[x['paid_nodes'] for x in ordered],'-o')
    counts.set_xlabel('AL step',fontsize=10);counts.set_ylabel('Paid MD nodes',fontsize=10);counts.grid(alpha=.2)
    counts.set_xlim(-.5,max(.5,step+.5));counts.set_ylim(0,max(1,max(x['paid_nodes'] for x in ordered)*1.2))
    counts.xaxis.set_major_locator(MaxNLocator(integer=True));counts.yaxis.set_major_locator(MaxNLocator(integer=True))
    rmse.tick_params(labelsize=9);counts.tick_params(labelsize=9)
    fig.suptitle(f'GAPS step {step:02d} | paid nodes: {len(paid)} | '+
                 f'external-FES diagnostic RMSE: {info["external_reference_RMSE_kbt"]:.4g} kBT',fontsize=12)
    if save_directory is not None:
        save_directory=Path(save_directory);save_directory.mkdir(parents=True,exist_ok=True)
        for suffix in ('png','pdf'):
            fig.savefig(save_directory/f'step_{step:02d}_single_seam_anchor.{suffix}',dpi=180,bbox_inches='tight')
    if not show:
        plt.close(fig)
    elif inline:
        from IPython.display import Image,display
        image=io.BytesIO();fig.savefig(image,format='png',dpi=110)
        display(Image(data=image.getvalue()))
        plt.close(fig)
    else:
        plt.show(block=False);fig.canvas.draw_idle();plt.pause(.05)
    print(f'[SPYDER STEP {step:02d}] paid={len(paid)}; '+
          f'external-FES diagnostic RMSE={info["external_reference_RMSE_kbt"]:.5g} kBT',flush=True)
    return True








if __name__=='__main__' and '--replot' in sys.argv:
    _single_replot_saved();raise SystemExit(0)

if __name__=='__main__' and not any(flag in sys.argv for flag in
        ('--run-batch','--cp2k-worker','--single-selftest','--original')):
    _single_prepare_spyder()

if __name__=='__main__' and '--run-batch' in sys.argv:
    _single_prepare_spyder(batch=True)

# Worker mode skips TensorFlow/GPflow imports entirely. All functions below this
# point remain part of this same Python file, including the original controller.
import argparse
import ast
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import time
import traceback
import numpy as np

HARTREE_KJ=2625.4996394799
BOHR_A=.529177210903


# COMPLETE JOINT-CV GEOMETRY AND MEAN-FORCE CODE
"""Second-order Cartesian AD and constrained-ensemble mean-force estimator.

Lengths are angstrom, potential forces kJ/mol/angstrom, angles radian.
The first two constraints are phi/psi; remaining constraints are base H bonds.
No finite differences or constraint-multiplier unit assumptions in production.
"""
import numpy as np


class Jet:
    def __init__(self, value, grad, hess):
        self.v, self.g, self.h = float(value), np.asarray(grad), np.asarray(hess)

    def constant(self, x):
        return Jet(x, np.zeros_like(self.g), np.zeros_like(self.h))

    def __add__(self, other):
        o = other if isinstance(other, Jet) else self.constant(other)
        return Jet(self.v+o.v, self.g+o.g, self.h+o.h)
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.g, -self.h)

    def __sub__(self, other): return self+-other
    def __rsub__(self, other): return other+-self

    def __mul__(self, other):
        o = other if isinstance(other, Jet) else self.constant(other)
        return Jet(self.v*o.v, self.g*o.v+o.g*self.v,
                   self.h*o.v+o.h*self.v+np.outer(self.g,o.g)+np.outer(o.g,self.g))
    __rmul__ = __mul__

    def unary(self, v, first, second):
        return Jet(v, first*self.g, first*self.h+second*np.outer(self.g,self.g))

    def __pow__(self, p):
        return self.unary(self.v**p, p*self.v**(p-1), p*(p-1)*self.v**(p-2))

    def __truediv__(self, other):
        return self*(other**-1 if isinstance(other, Jet) else 1./other)

    def exp(self):
        v=np.exp(self.v)
        return self.unary(v,v,v)


def atan2(y,x):
    r2=x.v*x.v+y.v*y.v
    if r2 < 1e-24: raise ValueError('Singular torsion')
    fx,fy=-y.v/r2,x.v/r2
    h=fx*x.h+fy*y.h
    h += (2*x.v*y.v/r2**2)*(np.outer(x.g,x.g)-np.outer(y.g,y.g))
    h += ((y.v*y.v-x.v*x.v)/r2**2)*(np.outer(x.g,y.g)+np.outer(y.g,x.g))
    return Jet(np.arctan2(y.v,x.v),fx*x.g+fy*y.g,h)


def variables(values):
    values=np.asarray(values).ravel();n=len(values);eye=np.eye(n)
    return [Jet(v,eye[i],np.zeros((n,n))) for i,v in enumerate(values)]


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def torsion_jet(x, ids, box):
    # Work with minimum-image bond vectors; the image is locally constant.
    pts=[x[3*i:3*i+3] for i in ids]
    def bond(a,b):
        d=[u-v for u,v in zip(a,b)]
        return [q-float(L*np.rint(q.v/L)) for q,L in zip(d,box)]
    b0,b1,b2=bond(pts[0],pts[1]),bond(pts[2],pts[1]),bond(pts[3],pts[2])
    norm=dot(b1,b1)**.5;b1=[q/norm for q in b1]
    s0,s2=dot(b0,b1),dot(b2,b1)
    v=[a-s0*b for a,b in zip(b0,b1)];w=[a-s2*b for a,b in zip(b2,b1)]
    return atan2(dot(cross(b1,v),w),dot(v,w))


def distance_jet(x,i,j,box):
    d=[x[3*i+k]-x[3*j+k] for k in range(3)]
    d=[q-float(L*np.rint(q.v/L)) for q,L in zip(d,box)]
    return dot(d,d)**.5


def molecular_jets(xyz, model):
    x=variables(xyz)
    jets=[torsion_jet(x,ids,model['box_angstrom']) for ids in model['torsions_zero_based']]
    jets += [distance_jet(x,*bond['atoms'],model['box_angstrom']) for bond in model['h_bonds']]
    return jets


def normal_fields(jets, inverse_masses, ncv=2):
    J=np.array([q.g for q in jets]);H=np.array([q.h for q in jets])
    W=np.asarray(inverse_masses)
    Z=(J*W)@J.T
    sign,logdet=np.linalg.slogdet(Z)
    if sign<=0 or np.linalg.cond(Z)>1e12: raise ValueError('Singular constraint metric')
    inv=np.linalg.inv(Z);B=(inv@J)*W
    # dZ[k,l,a] = derivative of metric entry k,l in Cartesian direction a.
    dZ=np.einsum('kba,b,lb->kla',H,W,J,optimize=True)
    dZ=dZ+dZ.transpose(1,0,2)
    trace=np.einsum('kaa,a->k',H,W)
    divergence=inv[:ncv]@trace-np.einsum('ik,kla,la->i',inv[:ncv],dZ,B,optimize=True)
    if len(jets)>ncv:
        base=Z[ncv:,ncv:]
        sb,logbase=np.linalg.slogdet(base)
        if sb<=0:raise ValueError('Singular base-constraint metric')
        gradlogbase=np.einsum('lk,kla->a',np.linalg.inv(base),dZ[ncv:,ncv:],optimize=True)
    else:
        logbase=0.;gradlogbase=np.zeros(J.shape[1])
    correction=divergence+.5*(B[:ncv]@gradlogbase)
    np.testing.assert_allclose(J@B[:ncv].T,np.eye(len(jets))[:,:ncv],atol=1e-8)
    return B[:ncv],correction,.5*(logbase-logdet),J


def frame_estimator(xyz, forces_kj_angstrom, masses, model, temperature):
    jets=molecular_jets(xyz,model)
    B,corr,logw,J=normal_fields(jets,np.repeat(1./np.asarray(masses),3))
    kbt=.008314462618*temperature
    potential=-B@np.asarray(forces_kj_angstrom).ravel()
    return potential-kbt*corr,logw,np.array([q.v for q in jets]),potential,-kbt*corr


def selftest():
    """Independent quadrature of a nonlinear two-CV model with a base constraint.

    C=(x exp(.3z), y+.2z^2, w exp(.15z)), U=|r|^2/2.
    Integrate the constrained density along z, apply the estimator, compare
    against a finite derivative of the exact conditional partition integral.
    Tests coupling, determinant reweighting, base constraints, and entropy sign.
    """
    from scipy.integrate import quad
    a,b,c=.3,.2,.15
    W=1./np.array([1.,2.,3.,4.]);target=np.array([.7,-.4])
    def jets_at(q):
        x,y,z,w=variables(q)
        return [x*(a*z).exp(),y+b*z*z,w*(c*z).exp()]
    def point(t,z):return np.array([t[0]*np.exp(-a*z),t[1]-b*z*z,z,0.])
    def partition(t):
        return quad(lambda z:np.exp(-.5*np.sum(point(t,z)**2)-a*z),-9,9,epsabs=1e-10)[0]
    expected=np.array([-(np.log(partition(target+np.eye(2)[i]*1e-4))-
                         np.log(partition(target-np.eye(2)[i]*1e-4)))/2e-4 for i in range(2)])
    def terms(z):
        q=point(target,z);jets=jets_at(q)
        B,corr,lw,J=normal_fields(jets,W)
        # constrained surface density in this parameterization
        Z=(J*W)@J.T
        density=np.exp(-.5*np.sum(q*q)-(a+c)*z+.5*np.linalg.slogdet(Z)[1])
        weight=density*np.exp(lw)
        return weight,weight*(B@q-corr)
    denom=quad(lambda z:terms(z)[0],-9,9,epsabs=1e-9)[0]
    got=np.array([quad(lambda z:terms(z)[1][i],-9,9,epsabs=1e-9)[0]/denom for i in range(2)])
    np.testing.assert_allclose(got,expected,atol=2e-7,rtol=2e-7)
    # Check divergence against independent Cartesian finite differences.
    q=point(target,.4);B,corr,lw,J=normal_fields(jets_at(q),W)
    numerical=np.zeros(2)
    for j in range(4):
        step=np.eye(4)[j]*1e-5
        bp=normal_fields(jets_at(q+step),W)[0]
        bm=normal_fields(jets_at(q-step),W)[0]
        numerical+=(bp[:,j]-bm[:,j])/2e-5
    # Base metric term is .15*B_z at w=0.
    np.testing.assert_allclose(corr,numerical+c*B[:,2],atol=2e-8)
    return dict(exact_gradient=expected.tolist(),estimated_gradient=got.tolist(),
                nonlinear_two_cv_base_constraint_test='PASS')


# COMPLETE CP2K SAMPLING / ANALYSIS IMPLEMENTATION
def read(p):return json.loads(Path(p).read_text())


def write(p,obj):
    """Durable atomic JSON replacement; preserve the previous file on failure."""
    import tempfile
    p=Path(p);payload=json.dumps(obj,indent=2,allow_nan=False)+'\n'
    fd,name=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    tmp=Path(name)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        tmp.unlink(missing_ok=True)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wrapped(d):return (np.asarray(d)+np.pi)%(2*np.pi)-np.pi


def log(message):print(message,flush=True)


def run(command,cwd,filename):
    import tempfile,gzip
    cwd=Path(cwd).resolve()
    required=('cp2k.inp','alanine.prmtop','start.xyz')
    for name in required:
        if not (cwd/name).is_file():raise FileNotFoundError(cwd/name)
    local=Path(tempfile.mkdtemp(prefix='gaps_single_',dir='/tmp'))
    if shutil.disk_usage(local).free<512*1024**2:
        shutil.rmtree(local)
        raise RuntimeError('Insufficient node-local temporary disk space')
    try:
        for name in required:shutil.copy2(cwd/name,local/name)
        write(cwd/'local_execution.json',dict(host=os.uname().nodename,temporary_directory=str(local),started=time.asctime()))
        with (local/filename).open('w') as output:
            result=subprocess.run(list(map(str,command)),cwd=local,stdout=output,stderr=subprocess.STDOUT)
        # Large MD logs stay node-local until compressed. Static-check output
        # remains plain text because its parser consumes it immediately.
        if result.returncode==0 and (local/'md.out').exists():
            with (local/'md.out').open('rb') as src,gzip.open(local/'md.out.gz','wb',compresslevel=3) as dst:
                shutil.copyfileobj(src,dst)
            (local/'md.out').unlink()
        for item in local.iterdir():
            if item.is_file() and item.name not in required:shutil.copy2(item,cwd/item.name)
        if result.returncode:
            tail='\n'.join((cwd/filename).read_text(errors='replace').splitlines()[-45:])
            raise RuntimeError(f'CP2K failed ({result.returncode}): {command}\n{tail}')
    except BaseException:
        log(f'Local diagnostic files retained at {local}')
        raise
    else:shutil.rmtree(local)


def parse_force(text,natoms):
    rows=[];blocks=[]
    for line in text.splitlines():
        if 'FORCES| Atomic forces' in line:rows=[]
        w=line.split()
        if len(w)>=6 and w[0]=='FORCES|' and w[1].isdigit():
            if int(w[1])!=len(rows)+1:raise ValueError('CP2K force atom order')
            rows.append([float(v.replace('D','E')) for v in w[2:5]])
            if len(rows)==natoms:blocks.append(np.array(rows))
    if not blocks:raise ValueError('No complete static force block')
    return blocks[-1]*HARTREE_KJ/BOHR_A


def xyz_frames(path):
    """Return (step, title, labels, coordinates); never infer frame alignment."""
    frames=[]
    with Path(path).open() as f:
        while True:
            line=f.readline()
            if not line:break
            if not line.strip():continue
            n=int(line);title=f.readline().strip()
            match=re.search(r'\bi\s*=\s*(\d+)',title)
            if not match:raise ValueError(f'MD step missing from XYZ header: {title}')
            rows=[f.readline().split() for _ in range(n)]
            if any(len(row)<4 for row in rows):raise ValueError('Truncated XYZ')
            frames.append((int(match[1]),title,[r[0] for r in rows],
                           np.array([[float(v) for v in r[1:4]] for r in rows])))
    if not frames:raise ValueError(f'Empty trajectory {path}')
    steps=[f[0] for f in frames]
    if any(b<=a for a,b in zip(steps,steps[1:])):raise ValueError('Nonincreasing XYZ steps')
    return frames


def write_xyz(path,labels,xyz):
    Path(path).write_text(str(len(labels))+'\nstarting structure\n'+
                         ''.join(f'{a} {x:.12f} {y:.12f} {z:.12f}\n'
                                 for a,(x,y,z) in zip(labels,xyz)))


def static_input(template,coordinate='start.xyz',grid=64,project='static'):
    s=re.sub(r'(?m)^\s*PROJECT\s+.*$',f' PROJECT {project}',template,count=1)
    s=re.sub(r'(?m)^\s*GMAX\s+.*$',f'    GMAX {grid} {grid} {grid}',s,count=1)
    s=re.sub(r'(?m)^\s*COORD_FILE_FORMAT\s+.*$','   COORD_FILE_FORMAT XYZ',s,count=1)
    s=re.sub(r'(?m)^\s*COORD_FILE_NAME\s+.*$',f'   COORD_FILE_NAME {coordinate}',s,count=1)
    return s


def md_input(template,model,center,steps,seed,grid,dt,every):
    s=static_input(template,grid=grid,project='window')
    s=s.replace('RUN_TYPE ENERGY_FORCE','RUN_TYPE MD')
    s=s.replace('PRINT_LEVEL MEDIUM','PRINT_LEVEL LOW')
    # The high precision FORCE_EVAL output is only needed for static checks.
    s=re.sub(r'(?s)\n &PRINT\s*\n  &FORCES ON.*? &END PRINT','',s,count=1)
    cv='';cons=''
    specifications=[('TORSION',a,float(t),'rad') for a,t in
                    zip(model['torsions_zero_based'],center)]
    specifications += [('DISTANCE',b['atoms'],b['target_angstrom'],'angstrom')
                       for b in model['h_bonds']]
    for i,(kind,atoms,target,unit) in enumerate(specifications,1):
        cv+=f'  &COLVAR\n   &{kind}\n    ATOMS '+ ' '.join(str(a+1) for a in atoms)+f'\n   &END {kind}\n  &END COLVAR\n'
        cons+=f'  &COLLECTIVE\n   COLVAR {i}\n   INTERMOLECULAR TRUE\n   TARGET [{unit}] {target:.16g}\n  &END COLLECTIVE\n'
    if s.count('&END SUBSYS')!=1:raise ValueError('Unexpected template SUBSYS structure')
    s=s.replace('&END SUBSYS',cv+' &END SUBSYS')
    # Independent stochastic initializations; one executable per replica.
    s=s.replace('&END GLOBAL',f' SEED {seed}\n&END GLOBAL',1)
    s+=f'''\n&MOTION
 &MD
  ENSEMBLE NVT
  STEPS {steps}
  TIMESTEP [fs] {dt:.12g}
  TEMPERATURE 300
  &THERMOSTAT
   TYPE CSVR
   REGION GLOBAL
   &CSVR
    TIMECON [fs] 100
   &END CSVR
  &END THERMOSTAT
 &END MD
 &CONSTRAINT
  CONSTRAINT_INIT TRUE
  SHAKE_TOLERANCE 1.0E-10
{cons}  &LAGRANGE_MULTIPLIERS ON
   FILENAME =multipliers.dat
   COMMON_ITERATION_LEVELS 1
   &EACH
    MD {every}
   &END EACH
  &END LAGRANGE_MULTIPLIERS
 &END CONSTRAINT
 &PRINT
  &RESTART ON
   BACKUP_COPIES 0
   ADD_LAST NUMERIC
   &EACH
    MD 1000
   &END EACH
  &END RESTART
  &RESTART_HISTORY OFF
  &END RESTART_HISTORY
  &TRAJECTORY ON
   FORMAT XYZ
   UNIT angstrom
   FILENAME =trajectory.xyz
   &EACH
    MD {every}
   &END EACH
  &END TRAJECTORY
  &FORCES ON
   FORMAT XYZ
   UNIT hartree*bohr^-1
   FILENAME =forces.xyz
   &EACH
    MD {every}
   &END EACH
  &END FORCES
 &END PRINT
&END MOTION
'''
    return s


def ratio_mean(g,lw):
    w=np.exp(lw-np.max(lw));return np.sum(w[:,None]*g,axis=0)/w.sum()


def block_statistics(g,lw,blocks):
    n=len(g)
    if n<10*blocks:raise ValueError('Too few production samples for block estimates')
    chunks=np.array_split(np.arange(n),blocks)
    # Delete-block jackknife for a ratio estimator; keeps nonuniform metric weights.
    weights=np.exp(lw-lw.max());num=weights[:,None]*g
    den=weights.sum();total=num.sum(axis=0)
    loo=np.array([(total-num[i].sum(axis=0))/(den-weights[i].sum()) for i in chunks])
    cov=(blocks-1)/blocks*((loo-loo.mean(axis=0)).T@(loo-loo.mean(axis=0)))
    ess=float(weights.sum()**2/np.sum(weights**2))
    return total/den,cov,ess


def process_segment(rd,model,cfg,center):
    saved=rd/'estimator_samples.npz'
    if saved.exists():
        a=np.load(saved)
        return a['gradient'],a['log_weight'],read(rd/'geometry_report.json')
    xframes=xyz_frames(rd/'trajectory.xyz');fframes=xyz_frames(rd/'forces.xyz')
    if [x[0] for x in xframes]!=[x[0] for x in fframes]:raise ValueError('Coordinate/force step alignment failed')
    segment=read(rd/'segment.json')
    if xframes[-1][0]!=segment['steps']:raise ValueError('Trajectory stopped before requested final step')
    data=[];weights=[];pot=[];ent=[];maxang=0.;maxbond=0.
    for x,f in zip(xframes,fframes):
        if x[2]!=f[2]:raise ValueError('Coordinate/force atom labels differ')
        if x[0]<=cfg['equilibration_steps']:continue
        g,lw,cv,p,e=frame_estimator(x[3],f[3]*HARTREE_KJ/BOHR_A,model['masses_amu'],model,cfg['temperature_K'])
        maxang=max(maxang,float(np.max(np.abs(wrapped(cv[:2]-center)))))
        base=np.array([b['target_angstrom'] for b in model['h_bonds']])
        maxbond=max(maxbond,float(np.max(np.abs(cv[2:]-base))))
        data.append(g);weights.append(lw);pot.append(p);ent.append(e)
    if not data:raise ValueError('No production data')
    if np.degrees(maxang)>.002 or maxbond>1e-5:raise ValueError(f'Constraint drift: angles={np.degrees(maxang)}, bonds={maxbond}')
    data=np.array(data);weights=np.array(weights)
    if not np.isfinite(data).all() or not np.isfinite(weights).all():raise ValueError('Nonfinite estimator')
    report=dict(max_dihedral_error_degrees=float(np.degrees(maxang)),max_H_bond_error_angstrom=maxbond,
                production_frames=len(data),potential_mean_kj_mol_rad=np.mean(pot,axis=0).tolist(),
                entropy_mean_kj_mol_rad=np.mean(ent,axis=0).tolist())
    np.savez_compressed(saved,gradient=data,log_weight=weights,potential_gradient=pot,entropy_gradient=ent)
    write(rd/'geometry_report.json',report)
    return data,weights,report


def analyze_window(root,wid,cfg,settings):
    wd=Path(root)/'windows'/f'window_{wid:06d}'
    model=read(Path(root)/'cp2k_model/model.json');center=np.array(read(wd/'window.json')['center'])
    means=[];covs=[];reports=[];reasons=[];kbt=.008314462618*cfg['temperature_K']
    for rep in range(cfg['replicas_per_window']):
        segments=sorted((wd/f'rep_{rep:02d}').glob('segment_*/RUN_COMPLETE'))
        if not segments:raise ValueError('No complete replica segments')
        arrays=[process_segment(p.parent,model,cfg,center) for p in segments]
        g=np.concatenate([a[0] for a in arrays]);lw=np.concatenate([a[1] for a in arrays])
        mean,cov,ess=block_statistics(g,lw,settings['block_count'])
        _,coarse,_=block_statistics(g,lw,settings['block_count']//2)
        # Keep the more conservative covariance by diagonal inflation.
        inflation=max(1.,float(np.max(np.diag(coarse)/np.maximum(np.diag(cov),1e-20))))
        cov*=inflation
        half=len(g)//2;difference=ratio_mean(g[:half],lw[:half])-ratio_mean(g[half:],lw[half:])
        se=np.sqrt(np.diag(cov))
        if np.any(np.abs(difference)>4*np.sqrt(4*np.diag(cov))+.25*kbt):reasons.append(f'replica {rep} halves differ')
        if ess<.5*len(g):reasons.append(f'replica {rep} low metric-weight ESS')
        if inflation>4:reasons.append(f'replica {rep} block-error estimate unstable')
        means.append(mean);covs.append(cov)
        reports.append(dict(replica=rep,mean_gradient_kj_mol_rad=mean.tolist(),block_SE_kj_mol_rad=se.tolist(),
                            production_frames=len(g),metric_weight_ESS=ess,half_difference_kj_mol_rad=difference.tolist(),
                            segments=[a[2] for a in arrays],block_covariance_inflation=inflation))
    means=np.array(means);n=len(means);mean=means.mean(axis=0)
    cov=sum(covs)/(n*n)
    if n>1:
        cov+=np.cov(means,rowvar=False,ddof=1)/n
        diff=np.abs(means[0]-means[1]);noise=np.sqrt(np.diag(covs[0]+covs[1]))
        if np.any(diff>4*noise+.25*kbt):reasons.append('Independent replicas disagree')
    if np.any(np.sqrt(np.diag(cov))>.5*kbt):reasons.append('Mean-gradient SE exceeds 0.5 kBT/rad')
    if not (np.isfinite(mean).all() and np.isfinite(cov).all()):
        raise ValueError('Nonfinite mean gradient or covariance: invalid MD estimate')
    # Statistical convergence is diagnostic only. Physical validity was enforced in
    # process_segment/validate_segment_outputs, independently of this policy.
    summary=dict(window=wid,center_rad=center.tolist(),gradient_kj_mol_rad=mean.tolist(),covariance_kj2_mol2_rad2=cov.tolist(),
                 replicas=reports,quality_pass=True,quality_failures=[],statistical_warnings=reasons,
                 sampling_policy='one fixed-length production segment per replica; no SE-triggered extensions',
                 estimator='joint-CV Cartesian potential-force/divergence estimator; rigid-H-bond base measure; determinant reweighting',
                 note='Statistical warnings are recorded without extending MD. Fixed-time sampling does not prove ergodicity or global FES accuracy.')
    write(wd/'force_summary.json',summary)
    return summary


def estimate(root,wid,cfg,settings):
    wd=Path(root)/'windows'/f'window_{wid:06d}'
    result=read(wd/'force_summary.json')
    if not result['quality_pass']:raise RuntimeError(f'Window {wid} has not passed sampling checks')
    mean=np.array(result['gradient_kj_mol_rad']);cov=np.array(result['covariance_kj2_mol2_rad2'])
    rng=np.random.default_rng(settings['seed']+wid+100000)
    draws=rng.multivariate_normal(mean,cov,size=settings['bootstrap_draws'])
    return mean,draws


def static_force_check(root,rd,model):
    """Verify MD force output is the physical potential force, with correct units."""
    report=rd/'static_force_gate.json'
    if report.exists():return read(report)
    case=rd/'static_force_gate';case.mkdir(exist_ok=True)
    xf=xyz_frames(rd/'trajectory.xyz')[-1];ff=xyz_frames(rd/'forces.xyz')[-1]
    if xf[0]!=ff[0]:raise ValueError('Static gate frame mismatch')
    write_xyz(case/'start.xyz',xf[2],xf[3])
    shutil.copy2(root/'cp2k_model/alanine.prmtop',case/'alanine.prmtop')
    template=(root/'cp2k_model/template.inp').read_text()
    cfg=read(root/'campaign_config.json')
    (case/'cp2k.inp').write_text(static_input(template,grid=cfg['fft_grid']))
    run([os.environ['CP2K_BIN'],'-i','cp2k.inp','-o','static.out'],case,'console.out')
    physical=parse_force((case/'static.out').read_text(),len(model['masses_amu']))
    observed=ff[3]*HARTREE_KJ/BOHR_A
    error=observed-physical
    gate=dict(force_rms_kj_mol_angstrom=float(np.sqrt(np.mean(error**2))),max_component_kj_mol_angstrom=float(np.abs(error).max()))
    # MD XYZ precision limits this comparison; a wrong force unit or added
    # constraint force would exceed this tolerance by orders of magnitude.
    if gate['force_rms_kj_mol_angstrom']>.03 or gate['max_component_kj_mol_angstrom']>.15:
        raise ValueError(f'MD output is not consistent with static potential forces: {gate}')
    write(report,gate);return gate


def validate_segment_outputs(rd,expected_steps):
    """Read plain or gzip CP2K logs and validate both Cartesian streams."""
    import gzip
    rd=Path(rd);path=rd/'md.out'
    if not path.exists():path=rd/'md.out.gz'
    opener=gzip.open if path.suffix=='.gz' else open
    ended=aborted=False
    with opener(path,'rt',errors='replace') as f:
        for line in f:
            ended=ended or 'PROGRAM ENDED AT' in line
            aborted=aborted or 'ABORT' in line
    if not ended:raise ValueError('CP2K did not end normally')
    if aborted:raise ValueError('CP2K abort reported')
    x=xyz_frames(rd/'trajectory.xyz');forces=xyz_frames(rd/'forces.xyz')
    if x[-1][0]!=expected_steps:raise ValueError('Incomplete MD trajectory')
    if [frame[0] for frame in x]!=[frame[0] for frame in forces]:
        raise ValueError('Coordinate/force step alignment failed')
    if any(a[2]!=b[2] for a,b in zip(x,forces)):raise ValueError('Coordinate/force atom labels differ')
    return dict(final_step=int(x[-1][0]),frames=len(x),log_file=path.name)


def run_replica_segment(root,wid,rep,attempt):
    cfg=read(root/'campaign_config.json');model=read(root/'cp2k_model/model.json')
    wd=root/'windows'/f'window_{wid:06d}';meta=read(wd/'window.json')
    rd=wd/f'rep_{rep:02d}'/f'segment_{attempt:02d}';rd.mkdir(parents=True,exist_ok=True)
    if attempt != 0:raise ValueError('Fixed-cost protocol allows one production segment per replica')
    steps=cfg['equilibration_steps']+cfg['production_steps']
    seed=int(271800000+10000*wid+1009*rep+173*attempt+17)
    if (rd/'RUN_COMPLETE').exists():return rd
    if (rd/'md.out').exists() or (rd/'md.out.gz').exists():
        try:
            saved=read(rd/'segment.json')
            if saved['steps']!=steps or saved['seed']!=seed or saved['equilibration_steps']!=cfg['equilibration_steps']:
                raise ValueError('Saved segment protocol differs')
            recovery=validate_segment_outputs(rd,steps)
        except (OSError,ValueError,KeyError,EOFError) as exc:
            log(f'[RECOVERY] W{wid:06d} replica {rep} segment {attempt}: incomplete output preserved: {exc}')
        else:
            write(rd/'recovered_completion.json',dict(recovery,time=time.asctime(),reason='normal MD output recovered after compressed-log reader error'))
            (rd/'RUN_COMPLETE').write_text('Completed MD recovered; estimator check still required.\n')
            log(f'[MD REUSE SEGMENT] W{wid:06d} replica {rep} segment {attempt}: {steps} completed steps; no new MD')
            return rd
    if (rd/'md.out').exists() or (rd/'md.out.gz').exists():
        # Preserve interrupted diagnostics rather than mixing appended XYZs.
        archive=rd.with_name(rd.name+'_interrupted_'+str(time.time_ns()))
        rd.rename(archive);rd.mkdir()
    if attempt:
        prior=wd/f'rep_{rep:02d}'/f'segment_{attempt-1:02d}'/'trajectory.xyz'
    elif (wd/'initialization_pilots'/f'rep_{rep:02d}'/'start_after_pilot.xyz').exists():
        prior=None
    elif (wd/'prepared_start.xyz').exists():
        prior=None
    elif meta['parent_window'] is not None:
        parent=root/'windows'/f"window_{meta['parent_window']:06d}"/'rep_00'
        prior=sorted(parent.glob('segment_*/RUN_COMPLETE'))[-1].parent/'trajectory.xyz'
    else:prior=None
    if prior is not None:
        frame=xyz_frames(prior)[-1];write_xyz(rd/'start.xyz',frame[2],frame[3])
    elif (wd/'initialization_pilots'/f'rep_{rep:02d}'/'start_after_pilot.xyz').exists():
        shutil.copy2(wd/'initialization_pilots'/f'rep_{rep:02d}'/'start_after_pilot.xyz',rd/'start.xyz')
    elif (wd/'prepared_start.xyz').exists():shutil.copy2(wd/'prepared_start.xyz',rd/'start.xyz')
    else:shutil.copy2(root/'cp2k_model/start.xyz',rd/'start.xyz')
    shutil.copy2(root/'cp2k_model/alanine.prmtop',rd/'alanine.prmtop')
    (rd/'cp2k.inp').write_text(md_input((root/'cp2k_model/template.inp').read_text(),model,meta['center'],
                                       steps,seed,cfg['fft_grid'],cfg['timestep_ps']*1000,cfg['sample_every']))
    write(rd/'segment.json',dict(steps=steps,seed=seed,equilibration_steps=cfg['equilibration_steps']))
    started=time.monotonic()
    run([os.environ['CP2K_BIN'],'-i','cp2k.inp','-o','md.out'],rd,'console.out')
    validate_segment_outputs(rd,steps)
    (rd/'RUN_COMPLETE').write_text('Completed MD; estimator check still required.\n')
    write(rd/'timing.json',dict(seconds=time.monotonic()-started,steps=steps))
    return rd


def compact_accepted_window(wd):
    """Compact only accepted, locally owned windows; keep estimator samples.

    Keep final coordinate frames at their existing paths for the original
    parent-start logic, compact statistical samples, summaries and restarts.
    Raw Cartesian trajectories cannot be regenerated from the compact samples.
    """
    import gzip
    wd=Path(wd)
    if wd.is_symlink():return
    summary=wd/'force_summary.json'
    if not summary.exists() or not read(summary).get('quality_pass'):return
    nrep=int(read(wd.parent.parent/'campaign_config.json')['replicas_per_window'])
    reps=[wd/f'rep_{r:02d}' for r in range(nrep)]
    if not all((r/'COMPLETE').is_file() for r in reps):return
    freed=0;segments=0
    for rep in reps:
        if rep.is_symlink():continue
        for rd in sorted(rep.glob('segment_*')):
            if rd.is_symlink() or not (rd/'RUN_COMPLETE').exists():continue
            if not (rd/'estimator_samples.npz').exists() or not (rd/'geometry_report.json').exists():continue
            # Verify compact estimator arrays before discarding the raw input.
            with np.load(rd/'estimator_samples.npz',allow_pickle=False) as a:
                if len(a['gradient'])<1 or not np.isfinite(a['gradient']).all() or not np.isfinite(a['log_weight']).all():
                    raise ValueError('Invalid compact estimator samples: '+str(rd))
            trajectory=rd/'trajectory.xyz'
            if not trajectory.is_file():raise FileNotFoundError(trajectory)
            # Stream through frames: bounded memory and exact original precision.
            last=None
            with trajectory.open() as f:
                while True:
                    line=f.readline()
                    if not line:break
                    if not line.strip():continue
                    n=int(line);frame=[line,f.readline()]+[f.readline() for _ in range(n)]
                    if any(not line for line in frame) or any(len(line.split())<4 for line in frame[2:]):
                        raise ValueError('Truncated accepted trajectory: '+str(trajectory))
                    last=''.join(frame)
            if last is None:raise ValueError('Empty accepted trajectory')
            before=trajectory.stat().st_size
            tmp=trajectory.with_name('trajectory.compact.tmp')
            try:
                with tmp.open('w') as f:f.write(last);f.flush();os.fsync(f.fileno())
                os.replace(tmp,trajectory)
            finally:tmp.unlink(missing_ok=True)
            freed+=before-trajectory.stat().st_size
            for name in ('forces.xyz','multipliers.dat'):
                raw=rd/name
                if raw.is_file() and not raw.is_symlink():
                    freed+=raw.stat().st_size;raw.unlink()
            # Retain complete logs, compressed; no truncation of diagnostics.
            for name in ('md.out','cp2k_console.out'):
                raw=rd/name
                if not raw.is_file() or raw.is_symlink():continue
                zipped=rd/(name+'.gz');tmp=zipped.with_name(zipped.name+'.tmp')
                try:
                    with raw.open('rb') as src,gzip.open(tmp,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
                    os.replace(tmp,zipped)
                finally:tmp.unlink(missing_ok=True)
                freed+=raw.stat().st_size-zipped.stat().st_size;raw.unlink()
            segments+=1
    _compact_pilot_diagnostics(wd)
    write(wd/'storage_cleanup.json',dict(policy='accepted windows only; compact estimator samples and final coordinates retained',
          raw_Cartesian_reanalysis_available=False,segments_checked=segments,bytes_freed_this_pass=freed,time=time.asctime()))
    if freed:log(f'[STORAGE] {wd.name}: freed {freed/1024**2:.1f} MiB; estimator samples retained')


def constrained_start_minimize(evaluate,constraints,x0,force_tolerance=10.,max_iterations=250,max_evaluations=600):
    """Projected Cartesian BFGS with exact nonlinear-constraint retraction.

    evaluate returns physical energy [kJ/mol] and Cartesian forces [kJ/mol/A].
    constraints returns residuals and the exact Cartesian Jacobian. No penalty
    potential is added. This prepares coordinates; it supplies no force labels.
    """
    x=np.asarray(x0,float).ravel().copy();n=len(x);calls=0;history=[]
    def retract(y):
        y=y.copy()
        for _ in range(30):
            c,J=constraints(y)
            if not np.isfinite(c).all() or not np.isfinite(J).all():raise ValueError('Nonfinite initialization constraint')
            if np.max(np.abs(c))<1.e-10:return y,J
            y-=J.T@np.linalg.solve(J@J.T,c)
        raise ValueError('Exact initialization constraint projection failed')
    def physical(y):
        nonlocal calls
        if calls>=max_evaluations:raise _InitializationEvaluationLimit('Initialization force-evaluation limit reached')
        calls+=1;e,f=evaluate(y)
        if not np.isfinite(e) or not np.isfinite(f).all():raise ValueError('Nonfinite initialization energy/force')
        return float(e),-np.asarray(f,float).ravel()
    def tangent(v,J):return v-J.T@np.linalg.solve(J@J.T,J@v)
    x,J=retract(x);energy,gradient=physical(x);initial_energy=energy
    inverse=np.eye(n)/1000.
    for iteration in range(max_iterations+1):
        g=tangent(gradient,J)
        maximum=float(np.max(np.linalg.norm(g.reshape(-1,3),axis=1)))
        history.append(dict(iteration=iteration,energy_kj_mol=energy,max_tangent_force_kj_mol_angstrom=maximum,evaluations=calls))
        if maximum<=force_tolerance:
            return x,dict(converged=True,initial_energy_kj_mol=initial_energy,final_energy_kj_mol=energy,
                          max_tangent_force_kj_mol_angstrom=maximum,constraint_residual=float(np.max(np.abs(constraints(x)[0]))),
                          iterations=iteration,evaluations=calls,history=history,
                          method='physical-force projected BFGS with exact constraint retraction; no springs')
        if iteration==max_iterations:break
        direction=tangent(-inverse@g,J)
        if g@direction>=-1.e-12:
            inverse=np.eye(n)/1000.;direction=-g/1000.
        norm=float(np.max(np.linalg.norm(direction.reshape(-1,3),axis=1)))
        if norm<1.e-14:raise ValueError('Initialization minimizer stalled')
        if norm>.03:direction*=.03/norm
        accepted=False;alpha=1.
        for _ in range(16):
            try:
                candidate,Jnew=retract(x+alpha*direction)
                candidate_energy,candidate_gradient=physical(candidate)
            except _InitializationEvaluationLimit:raise
            except (RuntimeError,ValueError,np.linalg.LinAlgError):
                alpha*=.5;continue
            displacement=candidate-x
            if candidate_energy<=energy+1.e-4*float(g@displacement):
                accepted=True;break
            alpha*=.5
        if not accepted:raise RuntimeError('Initialization minimizer could not find a physical downhill constrained step')
        new_g=tangent(candidate_gradient,Jnew);y=new_g-g;sy=float(displacement@y)
        if sy>1.e-10*np.linalg.norm(displacement)*np.linalg.norm(y):
            rho=1./sy;v=np.eye(n)-rho*np.outer(displacement,y)
            inverse=v@inverse@v.T+rho*np.outer(displacement,displacement)
        x,J,energy,gradient=candidate,Jnew,candidate_energy,candidate_gradient
    raise RuntimeError(f'Constrained initialization did not converge in {max_iterations} iterations; final tangent force={maximum:.6g} kJ/mol/A')


class _InitializationEvaluationLimit(RuntimeError):pass


def _initialization_coordinates(path):
    lines=Path(path).read_text().splitlines();n=int(lines[0]);rows=[s.split() for s in lines[2:2+n]]
    if len(rows)!=n or any(len(row)<4 for row in rows):raise ValueError('Truncated initialization XYZ')
    labels=[row[0] for row in rows];xyz=np.array([[float(v) for v in row[1:4]] for row in rows])
    return labels,xyz


def _initialization_constraint_function(model,center):
    target=np.r_[np.asarray(center,float),[bond['target_angstrom'] for bond in model['h_bonds']]]
    def constraints(flat):
        jets=molecular_jets(np.asarray(flat).reshape(-1,3),model)
        residual=np.array([q.v for q in jets])-target;residual[:2]=wrapped(residual[:2])
        return residual,np.array([q.g for q in jets])
    return constraints


def _relax_initialization(root,wd,cfg,model,center):
    import tempfile
    destination=wd/'initialization_relaxation';destination.mkdir(exist_ok=True)
    finished=destination/'report.json';output=destination/'relaxed_start.xyz'
    if finished.exists() and read(finished).get('converged') and output.exists():return output
    labels,xyz=_initialization_coordinates(wd/'prepared_start.xyz')
    constraints=_initialization_constraint_function(model,center)
    local=Path(tempfile.mkdtemp(prefix='gaps_relax_',dir='/tmp'))
    started=time.monotonic();counter=0;last_tail=''
    log(f'[INIT RELAX] {wd.name}: physical-energy minimization at fixed phi/psi and H bonds; no production labels')
    try:
        shutil.copy2(root/'cp2k_model/alanine.prmtop',local/'alanine.prmtop')
        template=(root/'cp2k_model/template.inp').read_text()
        (local/'cp2k.inp').write_text(static_input(template,grid=cfg['fft_grid'],project='relax'))
        def evaluate(flat):
            nonlocal counter,last_tail
            counter+=1;write_xyz(local/'start.xyz',labels,np.asarray(flat).reshape(-1,3))
            out=local/'relax.out';out.unlink(missing_ok=True)
            with (local/'console.out').open('w') as log_file:
                result=subprocess.run([os.environ['CP2K_BIN'],'-i','cp2k.inp','-o','relax.out'],cwd=local,stdout=log_file,stderr=subprocess.STDOUT)
            text=out.read_text(errors='replace') if out.exists() else (local/'console.out').read_text(errors='replace')
            last_tail='\n'.join(text.splitlines()[-60:])
            if result.returncode or 'PROGRAM ENDED AT' not in text or '[ABORT]' in text:
                raise RuntimeError('Physical initialization force evaluation failed: '+last_tail)
            energy_lines=[line for line in text.splitlines() if 'ENERGY|' in line and 'Total FORCE_EVAL' in line and 'energy' in line]
            if not energy_lines:raise ValueError('Initialization energy line missing')
            energy=float(energy_lines[-1].split()[-1].replace('D','E'))*HARTREE_KJ
            if counter%10==0:log(f'[INIT RELAX] {wd.name}: force evaluation {counter}; energy={energy:.3f} kJ/mol')
            return energy,parse_force(text,len(labels)).ravel()
        result,report=constrained_start_minimize(evaluate,constraints,xyz.ravel())
        write_xyz(output,labels,result.reshape(-1,3))
        report.update(center_rad=list(map(float,center)),source=str(wd/'prepared_start.xyz'),
                      seconds=time.monotonic()-started,training_label=False)
        write(finished,report)
        log(f'[INIT RELAXED] {wd.name}: energy {report["initial_energy_kj_mol"]:.3f} -> {report["final_energy_kj_mol"]:.3f} kJ/mol; {report["evaluations"]} physical force evaluations')
    except BaseException as exc:
        write(destination/'failure.json',dict(error=str(exc),force_evaluations=counter,local_directory=str(local),seconds=time.monotonic()-started))
        (destination/'last_force_evaluation.out').write_text(last_tail)
        log(f'Initialization diagnostics retained at {local}')
        raise
    else:shutil.rmtree(local)
    return output


class _PilotPhysicsFailure(ValueError):pass


def _pilot_constraint_values(xyz,model):
    """Fast coordinate-only checks; production's exact AD estimator is unchanged."""
    xyz=np.asarray(xyz,float);box=np.asarray(model['box_angstrom'],float)
    def delta(a,b):
        d=xyz[a]-xyz[b];return d-box*np.rint(d/box)
    values=[]
    for a,b,c,d in model['torsions_zero_based']:
        b0=delta(a,b);b1=delta(c,b);b2=delta(d,c)
        length=np.linalg.norm(b1)
        if length<1.e-12:raise _PilotPhysicsFailure('Degenerate pilot torsion bond')
        b1=b1/length;v=b0-np.dot(b0,b1)*b1;w=b2-np.dot(b2,b1)*b1
        if min(np.linalg.norm(v),np.linalg.norm(w))<1.e-12:raise _PilotPhysicsFailure('Degenerate pilot torsion plane')
        values.append(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w)))
    values.extend(float(np.linalg.norm(delta(*bond['atoms']))) for bond in model['h_bonds'])
    return np.array(values)


def _pilot_input(template,model,center,steps,seed,grid,dt_fs):
    # Dense output is restricted to short pilots. Production sampling is unchanged.
    text=md_input(template,model,center,steps,seed,grid,dt_fs,1)
    velocities='''  &VELOCITIES ON
   FORMAT XYZ
   UNIT angstrom*fs^-1
   FILENAME =velocities.xyz
   &EACH
    MD 1
   &END EACH
  &END VELOCITIES
'''
    marker=' &PRINT\n  &RESTART ON'
    if text.count(marker)!=1:raise ValueError('Unexpected pilot print section')
    return text.replace(marker,' &PRINT\n'+velocities+'  &RESTART ON',1)


def _assess_dense_pilot(rd,model,center,steps,dt_ps):
    import gzip
    validate_segment_outputs(rd,steps)
    path=rd/'md.out';path=path if path.exists() else rd/'md.out.gz'
    opener=gzip.open if path.suffix=='.gz' else open
    temperatures=[];warnings=[];shake=0;rattle=0
    with opener(path,'rt',errors='replace') as f:
        for line in f:
            low=line.lower()
            if 'not converged' in low:
                if 'shake' in low:shake+=1
                if 'rattle' in low:rattle+=1
                warnings.append(line.strip())
            if 'MD|' in line and 'Temperature [K]' in line:
                fields=line.split('Temperature [K]',1)[1].split()
                try:temperatures.append(float(fields[0].replace('D','E')))
                except (ValueError,IndexError):raise _PilotPhysicsFailure('Pilot temperature overflow/missing')
    if not temperatures or not np.isfinite(temperatures).all() or min(temperatures)<=0 or max(temperatures)>1500:
        raise _PilotPhysicsFailure('Pilot temperature invalid or above 1500 K')
    frames=xyz_frames(rd/'trajectory.xyz');velocity=xyz_frames(rd/'velocities.xyz')
    positions={frame[0]:frame for frame in frames};velocities={frame[0]:frame for frame in velocity}
    required=set(range(1,steps+1))
    if not required.issubset(positions) or not required.issubset(velocities):
        raise ValueError('Dense pilot positions and velocities must cover every MD step')
    target=np.r_[np.asarray(center,float),[b['target_angstrom'] for b in model['h_bonds']]]
    errors=[];rates=[]
    for step in sorted(positions):
        frame=positions[step];x=frame[3]
        if not np.isfinite(x).all():raise _PilotPhysicsFailure('Nonfinite pilot coordinates')
        error=_pilot_constraint_values(x,model)-target;error[:2]=wrapped(error[:2]);errors.append(error)
        if step not in required:continue
        vframe=velocities[step]
        if frame[2]!=vframe[2]:raise ValueError('Pilot position/velocity atom labels differ')
        v=vframe[3]
        if not np.isfinite(v).all():raise _PilotPhysicsFailure('Nonfinite pilot velocity')
        # Directional derivative is diagnostic only, in rad/fs and A/fs.
        # Wrapping the angular difference handles phi/psi at the periodic seam.
        epsilon_fs=1.e-3
        difference=(_pilot_constraint_values(x+epsilon_fs*v,model)-
                    _pilot_constraint_values(x-epsilon_fs*v,model))
        difference[:2]=wrapped(difference[:2]);rates.append(difference/(2*epsilon_fs))
    error=np.asarray(errors);rate=np.asarray(rates)
    angles=np.degrees(np.max(np.abs(error[:,:2]),axis=0));bond=float(np.max(np.abs(error[:,2:])))
    angular_rate=float(np.max(np.abs(rate[:,:2])));bond_rate=float(np.max(np.abs(rate[:,2:])))
    report=dict(steps=steps,timestep_ps=dt_ps,every_step_checked=True,velocity_units='angstrom/fs',
                maximum_temperature_K=max(temperatures),max_dihedral_error_degrees=angles.tolist(),max_bond_error_angstrom=bond,
                max_angle_velocity_error_rad_fs=angular_rate,max_bond_velocity_error_angstrom_fs=bond_rate,
                SHAKE_warning_count=shake,RATTLE_warning_count=rattle,warning_examples=warnings[:5],
                warning_policy='record warnings; require measured position, velocity, temperature and completion checks',training_label=False)
    write(rd/'measured_checks.json',report)
    np.savez_compressed(rd/'constraint_diagnostics.npz',position_errors=error,velocity_errors_per_fs=rate,
                        temperatures_K=temperatures,center_rad=center)
    if np.max(angles)>.002 or bond>1.e-5:raise _PilotPhysicsFailure('Measured pilot position constraints exceed tolerance')
    if angular_rate>1.e-6 or bond_rate>1.e-6:raise _PilotPhysicsFailure('Measured pilot velocity constraints exceed tolerance')
    report['passed']=True
    return frames[-1],report


def _pilot_phase(root,wd,rep,cfg,model,center,start,directory,steps,dt_ps,seed):
    rd=Path(directory);rd.mkdir(parents=True,exist_ok=True)
    text=_pilot_input((root/'cp2k_model/template.inp').read_text(),model,center,steps,seed,cfg['fft_grid'],dt_ps*1000)
    identity=hashlib.sha256(text.encode()+Path(start).read_bytes()+(root/'cp2k_model/alanine.prmtop').read_bytes()).hexdigest()
    output=rd/'start_after_pilot.xyz';report_file=rd/'report.json';failure=rd/'failure.json'
    if report_file.exists() and output.exists():
        report=read(report_file)
        if report.get('passed') and report.get('input_sha256')==identity:return output
    if failure.exists():
        saved=read(failure)
        if saved.get('input_sha256')==identity and saved.get('physical_failure'):
            raise _PilotPhysicsFailure('Saved failed pilot: '+saved['error'])
    if (rd/'md.out').exists() or (rd/'md.out.gz').exists():
        # Preserve incomplete/mismatched phases before an explicitly new run.
        archive=rd.with_name(rd.name+'_preserved_'+str(time.time_ns()));rd.rename(archive);rd.mkdir()
    shutil.copy2(start,rd/'start.xyz');shutil.copy2(root/'cp2k_model/alanine.prmtop',rd/'alanine.prmtop')
    (rd/'cp2k.inp').write_text(text);started=time.monotonic()
    write(rd/'phase_input.json',dict(input_sha256=identity,requested_steps=steps,timestep_ps=dt_ps,training_label=False))
    log(f'[INIT PILOT] {wd.name} replica {rep}: {steps} steps, dt={dt_ps*1000:g} fs; every-step diagnostics')
    try:
        try:run([os.environ['CP2K_BIN'],'-i','cp2k.inp','-o','md.out'],rd,'console.out')
        except RuntimeError as exc:
            path=rd/'md.out'
            text=path.read_text(errors='replace') if path.exists() else ''
            if 'GEOMETRY wrong' in text or 'NOT converged' in text:
                raise _PilotPhysicsFailure(str(exc)) from exc
            raise
        frame,report=_assess_dense_pilot(rd,model,center,steps,dt_ps)
    except _PilotPhysicsFailure as exc:
        write(failure,dict(input_sha256=identity,physical_failure=True,error=str(exc),requested_steps=steps,timestep_ps=dt_ps,training_label=False,seconds=time.monotonic()-started))
        raise
    write_xyz(output,frame[2],frame[3]);report.update(input_sha256=identity,seconds=time.monotonic()-started)
    write(report_file,report)
    log(f'[INIT PILOT PASS] {wd.name} replica {rep}: max T={report["maximum_temperature_K"]:.1f} K; warning counts SHAKE/RATTLE={report["SHAKE_warning_count"]}/{report["RATTLE_warning_count"]}; measured checks passed')
    return output


def _initialization_pilot(root,wd,rep,cfg,model,center,relaxed):
    rd=wd/'initialization_pilots'/f'rep_{rep:02d}';rd.mkdir(parents=True,exist_ok=True)
    finished=rd/'report.json';output=rd/'start_after_pilot.xyz'
    # Already accepted pilots and production are preserved, never reclassified.
    if finished.exists() and read(finished).get('passed') and output.exists():return output
    if (rd/'md.out').exists() or (rd/'md.out.gz').exists():
        archive=rd.with_name(rd.name+'_legacy_'+str(time.time_ns()));rd.rename(archive);rd.mkdir()
        write(rd/'legacy_pilot.json',dict(preserved_at=str(archive),reason='replace sampled warning-only check with dense measured gate'))
    baseline_dt=float(cfg['timestep_ps']);duration_ps=1000*baseline_dt
    seed=900000000+10000*int(read(wd/'window.json')['id'])+1009*rep
    def phase(name,start,dt_ps,phase_seed):
        steps=int(round(duration_ps/dt_ps))
        return _pilot_phase(root,wd,rep,cfg,model,center,start,rd/name,steps,dt_ps,phase_seed)
    failures=[]
    try:
        selected=phase('check_original_dt',relaxed,baseline_dt,seed)
    except _PilotPhysicsFailure as exc:
        failures.append(str(exc));selected=None
        for number,factor in enumerate((.5,.25),1):
            log(f'[INIT RETRY] {wd.name} replica {rep}: bounded {factor:g}x-timestep warmup, then original-timestep confirmation')
            try:
                warm=phase(f'warmup_{number}',relaxed,baseline_dt*factor,seed+number*101)
                selected=phase(f'confirm_{number}',warm,baseline_dt,seed+number*101+17)
                break
            except _PilotPhysicsFailure as failure:
                failures.append(str(failure))
        if selected is None:
            write(rd/'retry_report.json',dict(passed=False,failures=failures,production_started=False))
            raise _PilotPhysicsFailure('All bounded initialization retries failed: '+'; '.join(failures))
    selected_report=read(selected.parent/'report.json')
    if selected_report['timestep_ps']!=baseline_dt:raise RuntimeError('Original-timestep confirmation missing')
    shutil.copy2(selected,output)
    selected_report.update(selected_phase=str(selected.parent),earlier_physical_failures=failures,
                           production_timestep_ps=baseline_dt,production_timestep_changed=False)
    write(finished,selected_report)
    return output


def _compact_pilot_diagnostics(wd):
    import gzip
    for report in (Path(wd)/'initialization_pilots').rglob('report.json'):
        if not read(report).get('passed'):continue
        for name in ('trajectory.xyz','forces.xyz','velocities.xyz','multipliers.dat'):
            raw=report.parent/name
            if not raw.is_file() or raw.is_symlink():continue
            zipped=raw.with_name(raw.name+'.gz');tmp=zipped.with_name(zipped.name+'.tmp')
            try:
                with raw.open('rb') as src,gzip.open(tmp,'wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst)
                os.replace(tmp,zipped)
            finally:tmp.unlink(missing_ok=True)
            raw.unlink()



def ensure_relaxed_initialization(root,wid,cfg,model):
    wd=root/'windows'/f'window_{wid:06d}'
    # Accepted/finished sampling is not invalidated by a preparation change.
    if all((wd/f'rep_{rep:02d}/segment_00/RUN_COMPLETE').exists() for rep in range(cfg['replicas_per_window'])):return
    center=read(wd/'window.json')['center']
    relaxed=_relax_initialization(root,wd,cfg,model,center)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['replicas_per_window']) as pool:
        tasks=[pool.submit(_initialization_pilot,root,wd,rep,cfg,model,center,relaxed) for rep in range(cfg['replicas_per_window'])]
        for task in tasks:task.result()


def worker(root,wid):
    """One fixed-duration MD measurement per replica; GP/path integration unchanged.

    Accept valid constrained-MD estimates even when block/replica statistical
    diagnostics are noisy. Trajectory, force, geometry and physical checks
    remain mandatory; no automatic sampling extensions are allowed.
    """
    root=Path(root).resolve();cfg=read(root/'campaign_config.json');settings=read(root/'adapter_settings.json')
    wd=root/'windows'/f'window_{wid:06d}'
    if (wd/'force_summary.json').exists() and read(wd/'force_summary.json').get('quality_pass') and all(
        (wd/f'rep_{r:02d}'/'COMPLETE').exists() for r in range(cfg['replicas_per_window'])
    ):
        compact_accepted_window(wd)
        return
    model=read(root/'cp2k_model/model.json')
    ensure_relaxed_initialization(root,wid,cfg,model)
    policy=dict(window=wid,production_steps_per_replica=int(cfg['production_steps']),
                equilibration_steps_per_replica=int(cfg['equilibration_steps']),
                timestep_ps=float(cfg['timestep_ps']),replicas=int(cfg['replicas_per_window']),
                maximum_production_segments_per_replica=1,
                statistical_quality_diagnostics='reported, never used to extend MD',
                force_estimator='unchanged',time=time.asctime())
    write(wd/'fixed_cost_policy.json',policy)
    log(f'[FIXED COST] W{wid:06d}: {cfg["production_steps"]} production steps + '
        f'{cfg["equilibration_steps"]} equilibration steps per replica; '
        f'{cfg["replicas_per_window"]} replicas; one attempt')
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['replicas_per_window']) as pool:
        tasks=[pool.submit(run_replica_segment,root,wid,r,0) for r in range(cfg['replicas_per_window'])]
        paths=[task.result() for task in tasks]
    if wid==0:static_force_check(root,paths[0],model)
    summary=analyze_window(root,wid,cfg,settings)
    if not summary['quality_pass']:
        raise RuntimeError(f'W{wid:06d}: MD data invalid; no automatic extension')
    for r in range(cfg['replicas_per_window']):
        (wd/f'rep_{r:02d}'/'COMPLETE').write_text('Physically valid fixed-cost MD measurement; statistical diagnostics in summary.\n')
    compact_accepted_window(wd)
    log(f'[FIXED COST COMPLETE] W{wid:06d}: gradient={summary["gradient_kj_mol_rad"]}; '
        f'statistical_warnings={summary.get("statistical_warnings",[])}')


def molecular_derivative_test(bundle):
    import parmed as pmd
    model=read(bundle/'cp2k_model/model.json')
    mol=pmd.load_file(str(bundle/'cp2k_model/alanine.prmtop'),
                      xyz=str(bundle/'simulations_essentials/reference_start.gro'))
    x=np.array(mol.coordinates);jets=molecular_jets(x,model)
    W=np.repeat(1./np.array(model['masses_amu']),3)
    B,corr,lw,J=normal_fields(jets,W)
    # Independent finite differences for every CV gradient and the divergence.
    h=1e-5;div=np.zeros(2);gradbase=np.zeros(x.size);max_gradient_error=0.
    for a in range(x.size):
        e=np.eye(x.size)[a].reshape(x.shape)*h
        jp=molecular_jets(x+e,model);jm=molecular_jets(x-e,model)
        numeric=wrapped(np.array([p.v-m.v for p,m in zip(jp[:2],jm[:2])]))/(2*h)
        max_gradient_error=max(max_gradient_error,float(np.max(np.abs(numeric-J[:2,a]))))
        bp=normal_fields(jp,W)[0];bm=normal_fields(jm,W)[0]
        div+=(bp[:,a]-bm[:,a])/(2*h)
        def base_logdet(js):
            jj=np.array([q.g for q in js[2:]])
            return np.linalg.slogdet((jj*W)@jj.T)[1]
        gradbase[a]=(base_logdet(jp)-base_logdet(jm))/(2*h)
    np.testing.assert_allclose(corr,div+.5*B@gradbase,atol=2e-5,rtol=2e-5)
    if max_gradient_error>2e-6:raise ValueError('Torsion derivative finite-difference test failed')
    # PBC invariance when individual atoms are shifted by whole box lengths.
    shifted=x.copy();shifted[4]+=np.array(model['box_angstrom'])
    jshift=molecular_jets(shifted,model)
    np.testing.assert_allclose(wrapped(np.array([q.v for q in jshift[:2]])-np.array([q.v for q in jets[:2]])),0,atol=1e-10)
    np.testing.assert_allclose(np.array([q.g for q in jshift]),J,atol=1e-9)
    return dict(torsion_gradient_max_FD_error=max_gradient_error,molecular_divergence_and_PBC='PASS',
                original_H_bond_constraints=len(model['h_bonds']))


# =============================================================================
# APPENDED PARALLEL WINDOW DISPATCH / PERIODIC INITIALIZATION
# Each array task requests two CPU slots for its two MD replicas. The controller
# submits all newly paid path nodes together, then waits before the next query.
# =============================================================================
def _ala_rotate_to_target(xyz,model,targets):
    """Rotate downstream fragments, preserving covalent lengths and angles.

    This is initial-coordinate preparation, not a restraint or a label source.
    Bond connectivity is required; the alanine phi/psi bonds must not be cyclic.
    """
    xyz=np.array(xyz,float,copy=True);box=np.asarray(model['box_angstrom'])
    adjacency=[set() for _ in xyz]
    for a,b in model['bond_graph']:
        adjacency[a].add(b);adjacency[b].add(a)
    original=xyz.copy();visited={0};queue=[0]
    for a in queue:
        for b in sorted(adjacency[a]):
            if b in visited:continue
            d=original[b]-original[a];d-=box*np.rint(d/box)
            xyz[b]=xyz[a]+d;visited.add(b);queue.append(b)
    if len(visited)!=len(xyz):raise ValueError('Initial molecule is not covalently connected')
    lengths=np.array([np.linalg.norm(xyz[a]-xyz[b]) for a,b in model['bond_graph']])
    for ids,target in zip(model['torsions_zero_based'],targets):
        a,b,c,d=ids
        moving={c};queue=[c]
        for u in queue:
            for v in sorted(adjacency[u]):
                if {u,v}=={b,c} or v in moving:continue
                moving.add(v);queue.append(v)
        if a in moving or b in moving or d not in moving:
            raise ValueError('A torsion bond is cyclic or its downstream fragment is ambiguous')
        current=torsion_jet(variables(xyz),ids,box).v
        delta=float(wrapped(target-current));axis=xyz[c]-xyz[b];axis/=np.linalg.norm(axis)
        indices=np.array(sorted(moving),int);v=xyz[indices]-xyz[b]
        rotated=v*np.cos(delta)+np.cross(axis,v)*np.sin(delta)+np.outer(v@axis,axis)*(1-np.cos(delta))
        xyz[indices]=xyz[b]+rotated
    actual=np.array([torsion_jet(variables(xyz),ids,box).v for ids in model['torsions_zero_based']])
    np.testing.assert_allclose(wrapped(actual-targets),0,atol=1e-8)
    np.testing.assert_allclose([np.linalg.norm(xyz[a]-xyz[b]) for a,b in model['bond_graph']],lengths,atol=1e-8)
    return xyz


def _ala_prepare_window_start(root,wd,source_trajectory,targets,model):
    destination=wd/'prepared_start.xyz'
    if destination.exists():return
    if source_trajectory is not None:
        frame=xyz_frames(source_trajectory)[-1];labels,xyz=frame[2],frame[3]
    else:
        # Model XYZ has an ordinary comment rather than an MD step header.
        lines=(root/'cp2k_model/start.xyz').read_text().splitlines();n=int(lines[0])
        rows=[s.split() for s in lines[2:2+n]]
        labels=[r[0] for r in rows];xyz=np.array([[float(v) for v in r[1:4]] for r in rows])
    xyz=_ala_rotate_to_target(xyz,model,targets)
    write_xyz(destination,labels,xyz)
    write(wd/'initialization.json',dict(method='Exact internal bond rotations; no springs',
           source_trajectory=None if source_trajectory is None else str(source_trajectory),target_rad=list(targets)))


def _ala_window_accepted(root,wid):
    wd=root/'windows'/f'window_{int(wid):06d}';summary=wd/'force_summary.json'
    nrep=int(read(root/'campaign_config.json')['replicas_per_window'])
    return summary.exists() and read(summary).get('quality_pass') and all((wd/f'rep_{r:02d}/COMPLETE').exists() for r in range(nrep))


def _ala_parallel_dispatch(adapter,wids):
    """Submit an SGE array, then require an accepted estimate for every node."""
    import shlex
    root=adapter.root;wids=list(map(int,wids))
    if not wids:return
    batches=root/'md_batches';batches.mkdir(exist_ok=True)
    # A controller restart must not duplicate still-running array workers.
    for record in sorted(batches.glob('batch_*/batch.json')):
        old=read(record)
        check=subprocess.run(['qstat','-j',str(old['job_id'])],capture_output=True,text=True)
        if check.returncode==0:
            log(f'[PARALLEL MD] Rejoining existing array {old["job_id"]}')
            _ala_wait_array(adapter,str(old['job_id']),old['windows'])
    wids=[wid for wid in wids if not _ala_window_accepted(root,wid)]
    if not wids:return
    for wid in wids:
        wd=root/'windows'/f'window_{wid:06d}'
        # A stale failure from an earlier interrupted submission is not reused.
        (wd/'ARRAY_FAILED').unlink(missing_ok=True)
        (wd/'ARRAY_FINISHED').unlink(missing_ok=True)
        previous_log=wd/'controller_worker.log'
        if previous_log.exists():shutil.copy2(previous_log,wd/f'controller_worker.before_array_{time.time_ns()}.log')
    batch=batches/f'batch_{time.time_ns()}';batch.mkdir()
    (batch/'windows.txt').write_text(''.join(f'{w}\n' for w in wids))
    script=batch/'run_array.sh'
    script.write_text('''#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -q long
#$ -pe smp 2
#$ -N GAPS_ALA_MD
set -euo pipefail
module unload cp2k/2026.3 2>/dev/null || true
module load cp2k/2026.3
export LD_LIBRARY_PATH="/software/c/cp2k/src/cp2k-2026.1/tools/toolchain/install/elpa-2024.05.001/cpu/lib:${LD_LIBRARY_PATH:-}"
export CP2K_BIN="$(command -v cp2k.psmp)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
LIST='''+shlex.quote(str(batch/'windows.txt'))+'''
ROOT='''+shlex.quote(str(root))+'''
WID=$(sed -n "${SGE_TASK_ID}p" "$LIST")
test -n "$WID"
printf -v TAG '%06d' "$WID"
WD="$ROOT/windows/window_$TAG"
if '''+shlex.quote(sys.executable)+' -u '+shlex.quote(str(root/'runtime_single.py'))+''' --cp2k-worker "$ROOT" "$WID" > "$WD/controller_worker.log" 2>&1; then
    touch "$WD/ARRAY_FINISHED"
else
    code=$?
    printf '%s\n' "$code" > "$WD/ARRAY_FAILED"
    exit "$code"
fi
''')
    scheduler_logs=Path(SINGLE_PROJECT)/'GAPS_scheduler_logs';scheduler_logs.mkdir(exist_ok=True)
    command=['qsub','-terse','-V','-t',f'1-{len(wids)}','-tc',str(min(MAX_PARALLEL_WINDOWS,len(wids))),
             '-wd',str(Path(SINGLE_PROJECT)),'-o',str(scheduler_logs),'-e',str(scheduler_logs),str(script)]
    try:result=subprocess.run(command,cwd=root,capture_output=True,text=True,check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('MD array submission failed: '+str(exc.stdout)+'\n'+str(exc.stderr)) from exc
    match=re.match(r'\s*(\d+)',result.stdout)
    if not match:raise RuntimeError('Could not parse array job id: '+result.stdout)
    job=match[1]
    write(batch/'batch.json',dict(job_id=job,windows=wids,max_concurrent=min(MAX_PARALLEL_WINDOWS,len(wids)),
          requested_CPU_slots_per_task=2,submitted=time.asctime()))
    log(f'[PARALLEL MD] Submitted array {job}: {len(wids)} windows; up to {min(MAX_PARALLEL_WINDOWS,len(wids))} concurrent, {adapter.cfg["replicas_per_window"]} replica(s) per window')
    _ala_wait_array(adapter,job,wids)


def _ala_wait_array(adapter,job,wids):
    root=adapter.root;wids=list(map(int,wids))
    previous=None;queue_absent_since=None;last_progress=time.monotonic()
    try:
        while True:
            complete=[];failed=[]
            for wid in wids:
                wd=root/'windows'/f'window_{wid:06d}'
                if _ala_window_accepted(root,wid):complete.append(wid)
                elif (wd/'ARRAY_FAILED').exists():failed.append(wid)
            state=(tuple(complete),tuple(failed))
            pending=len(wids)-len(complete)-len(failed)
            if state!=previous:
                # Shared-file visibility can lag the scheduler. Allow a fresh
                # grace period after any newly visible completion or failure.
                queue_absent_since=None
                adapter.status('parallel constrained MD',array_job=job,windows=wids,
                               completed_windows=len(complete),failed_windows=failed,pending_windows=pending,
                               total_windows=len(wids),paid_nodes=len(adapter.state['node_windows']))
                log(f'[PARALLEL MD {job}] accepted {len(complete)}/{len(wids)}; failed={failed}; pending={pending}')
                previous=state
            # A failed window no longer cancels other running batch members.
            if pending==0:
                if failed:raise RuntimeError('Batch finished. Failed windows '+str(failed)+'. Other windows were allowed to finish; accepted results preserved. Inspect controller_worker.log files.')
                break
            check=subprocess.run(['qstat','-j',job],capture_output=True,text=True)
            scheduler_errors=[line.strip() for line in check.stdout.splitlines() if 'error reason' in line.lower()]
            if scheduler_errors:raise RuntimeError('CRC scheduler prevented MD launch: '+scheduler_errors[0])
            if check.returncode and ('do not exist' in (check.stdout+check.stderr).lower() or 'does not exist' in (check.stdout+check.stderr).lower()):
                now=time.monotonic()
                if queue_absent_since is None:
                    queue_absent_since=now
                    log(f'[FINALIZE MD {job}] Array left queue; allowing up to 300 seconds without progress for shared completion files to become visible')
                if now-queue_absent_since>=300:
                    # Recheck the actual accepted results after the scheduler
                    # call, before declaring missing outputs. Never infer a
                    # scientific pass from queue disappearance or a log line.
                    complete=[wid for wid in wids if _ala_window_accepted(root,wid)]
                    failed=[wid for wid in wids if wid not in complete and (root/'windows'/f'window_{wid:06d}'/'ARRAY_FAILED').exists()]
                    if len(complete)==len(wids):break
                    missing=sorted(set(wids)-set(complete)-set(failed))
                    raise RuntimeError(f'Array left the queue; failed windows={failed}; missing completion markers={missing}. Outputs preserved.')
            else:queue_absent_since=None
            if time.monotonic()-last_progress>=30:
                log(f'[WAIT MD {job}] accepted {len(complete)}/{len(wids)}; failed={failed}; pending={pending}')
                last_progress=time.monotonic()
            time.sleep(10)
    except KeyboardInterrupt:
        log(f'[PAUSED] MD array {job} continues; completed output is retained')
        raise
    except Exception:
        log(f'[BATCH STOP] Array {job} has not been cancelled by this controller; completed data is retained')
        raise
    log(f'[PARALLEL MD {job}] Entire batch accepted; returning to original GAPS fit/acquisition')


if __name__=='__main__' and '--single-selftest' in sys.argv:
    print(json.dumps(selftest(),indent=2));raise SystemExit(0)
if __name__=='__main__' and '--cp2k-worker' in sys.argv:
    flag=sys.argv.index('--cp2k-worker')
    worker(sys.argv[flag+1],int(sys.argv[flag+2]));raise SystemExit(0)

ALANINE_MODE = '--original' not in sys.argv

# ORIGINAL ATTACHED V18-V17 CODE WITH GUARDED ALANINE HOOKS
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Stage-2 GP W3 2D  (PIP / MD-poly version, plotting aligned to toy script)
#
# Results:
#   - Oracle 2D scatter + diag slice (2D/3D) + face slice (2D/3D)
#   - Oracle W3 2D scatter
#   - Optional full-grid LSQR diagnostic scatter
#   - Optional per-step LSQR paid-vs-full-grid parity
#   - Per-step: student μ/σ 2D scatter (with overlays)
#   - Per-step: μ/σ diag slice (2D/3D), μ face slice (2D/3D)
#   - Per-step: absolute error |pred - true| scatter + diag slice (2D/3D) + face slice (2D/3D)
#   - Per-step: extracted W3 on diag and face (2D/3D)
#   - Per-step: parity (train+val), RMSE trace, MST trace
#   - Summary RMSE vs iter, MST vs iter
# =============================================================================

import os
import tempfile
from pathlib import Path

def _setup_mpl_tex_cache_to_tmp():
    root = Path(tempfile.mkdtemp(prefix="mpl_tex_", dir="/tmp"))
    os.environ["MPLCONFIGDIR"]   = str(root / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(root / "xdg_cache")
    os.environ["XDG_CONFIG_HOME"]= str(root / "xdg_config")
    os.environ["TEXMFVAR"]       = str(root / "texmfvar")
    os.environ["TEXMFCONFIG"]    = str(root / "texmfconfig")
    os.environ["TEXMFCACHE"]     = str(root / "texmfcache")
    os.environ["TMPDIR"]         = "/tmp"
    for k in ("MPLCONFIGDIR","XDG_CACHE_HOME","XDG_CONFIG_HOME","TEXMFVAR","TEXMFCONFIG","TEXMFCACHE"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

_setup_mpl_tex_cache_to_tmp()

# =============================================================================
# Imports
# =============================================================================
import argparse
import csv
import heapq
import numpy as np
import matplotlib as mpl
# Use the narrow hyphen-minus in tick labels instead of the wide Unicode minus.
# This applies globally to axes and colorbar numeric ticks.
mpl.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    import gpflow
    from gpflow import utilities as gpf_util
except BaseException as exc:
    _single_startup_failed(exc)
    raise

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import lsqr
from matplotlib.ticker import MaxNLocator, FuncFormatter

# =============================================================================
# CLI options
# =============================================================================
DEFAULT_START_MODE    = "cold"         # warm/cold
DEFAULT_TARGET_MODE   = "umb"          # umb/residual
DEFAULT_SYMM_MODE     = "sorted"         # sorted/invariants/none
DEFAULT_KERNEL_MODE   = "plain"        # structured/plain
DEFAULT_LABEL_SOURCE  = "lsqr"  # pathint|lsqr|oracle_paid|oracle_queried

RUN_TAG = "UmbDirect"

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--start_mode", choices=["warm", "cold"], default=DEFAULT_START_MODE,
                   help="warm=copies previous GP params; cold=rebuild from scratch each step")
    p.add_argument("--target_mode", choices=["umb", "residual"], default=DEFAULT_TARGET_MODE,
                   help="umb=GP trains on anchored Umb; residual=GP trains on Umb-(W2sum-W2_anchor)")
    p.add_argument("--symmetry_mode", choices=["sorted", "invariants", "none"], default=DEFAULT_SYMM_MODE,
                   help="Permutation-invariant inputs (PIP spirit): sorted or symmetric invariants")
    p.add_argument("--kernel_mode", choices=["structured", "plain"], default=DEFAULT_KERNEL_MODE,
                   help="structured=(1D+1D+1D)+3D, plain=use KERNEL_SPEC")
    p.add_argument("--label_source", choices=["pathint", "lsqr", "oracle_paid", "oracle_queried"],
                   default=DEFAULT_LABEL_SOURCE,
                   help=("Label source for GP training: "
                         "pathint=on-the-fly path integration of exact forces on paid graph; "
                         "lsqr=LSQR from forces on paid graph; "
                         "oracle_paid=direct oracle energies on ALL paid nodes; "
                         "oracle_queried=direct oracle energies on QUERIED nodes only "
                         "(still pays path nodes, but excludes them from training)."))
    return p.parse_args()

# =============================================================================
# MD ORACLE PARAMETERS
# =============================================================================
GRAFT = "0.30"  # "0.15" or "0.30"

D12_MIN, D12_MAX = 6.04, 11.0
D13_MIN, D13_MAX = 6.04, 11.0
D23_MIN, D23_MAX = 6.04, 11.0

N3 = 50
PLOT_MAX = 100000000
SEED = 12345

DO_FULLGRID_LSQR_DIAGNOSTIC = True
PLOT_FULLGRID_LSQR_DIAGNOSTIC = False

LSQR_PRINT_EQUATION_STATS = True
DW_CLIP = 40.0

# Continuous edge-integration settings.
# These make LSQR/pathint integrate through in-between quadrature points
# instead of relying only on endpoint-averaged grid forces.
PATHINT_QUAD_ORDER = int(os.environ.get("PATHINT_QUAD_ORDER", 16))
PATHINT_FD_EPS = float(os.environ.get("PATHINT_FD_EPS", 1e-4))

LSQR_INTEGRATION_MODE = os.environ.get("LSQR_INTEGRATION_MODE", "continuous").lower()  # continuous|endpoint
LSQR_EDGE_QUAD_ORDER = int(os.environ.get("LSQR_EDGE_QUAD_ORDER", PATHINT_QUAD_ORDER))
LSQR_FD_EPS = float(os.environ.get("LSQR_FD_EPS", PATHINT_FD_EPS))
LSQR_EDGE_BATCH_SIZE = int(os.environ.get("LSQR_EDGE_BATCH_SIZE", 3000))

PARAMS = {
    "0.15": {
        "dimer": {
            "k":  1.525067170652243,
            "x0": 8.363446322328578,
            "C": [-1.851203176386485, 2.658258259143189e-01, -2.640932576942780e-02,
                  1.349865646591411e-03, -3.729456173796494e-05, 4.265841953601893e-07],
            "rin": 10.0, "rout": 12.0,
        },
        "trimer": {
            "k":  5.039639114081719e-02,
            "x0": 4.999985984055976e+01,
            "C": [ 5.606910952220617e+02, -1.714059495393181e+03, -1.036597547189558e+00,
                  -2.465375325002020e+03,  9.698122318213179e+02,  9.011895158565868e+02,
                  -7.262392333263364e+01,  7.398735000575322e+02, -5.287031738300126e+02,
                  -5.878008118858911e+02,  8.285603106331479e+00,  1.427961679585090e+02,
                  -6.798268695275270e+01, -2.255193567829562e+02,  9.115792931264546e+01],
            "Ri": 6.0, "Ro": 10.0,
        },
    },
    "0.30": {
        "dimer": {
            "k":  1.426598250344009,
            "x0": 8.189162153344407,
            "C": [ 2.914781927874392e+00, -1.009595164815141e+00,  1.870210969460044e-01,
                 -2.030955490318639e-02,  1.239669836304152e-03, -4.171931659803831e-05,
                  6.155231317677564e-07],
            "rin": 10.0, "rout": 12.0,
        },
        "trimer": {
            "k":  2.446835361961843e-01,
            "x0": 2.118153650543119e+01,
            "C": [ 6.674892424965560e+00, -7.684095412814079e+00,  2.393000139152252e+00,
                  1.001282066575281e+00,  2.701725383910903e+00, -7.018160213524727e-01,
                 -3.928487047117967e-01, -7.150042936617682e-02,  9.622682681857697e-02,
                  5.045340594110370e-02, -1.027943320237391e-03, -6.432621099307617e-03,
                  2.111046653110089e-02,  6.310019710739115e-03, -1.157370931056368e-03],
            "Ri": 6.0, "Ro": 10.0,
        },
    },
}

# =============================================================================
# ORACLE FUNCTIONS
# =============================================================================
def switch_poly(d, rin=10.0, rout=12.0):
    d = np.asarray(d, float)
    x = (d - rin) / (rout - rin)
    return np.where(x < 0.0, 1.0,
           np.where(x >= 1.0, 0.0, 1.0 + 2.0*(2.0*x - 3.0)*(x**2)))

def switch_cos2(d, Ri, Ro):
    d = np.asarray(d, float)
    t = (d - Ri) / (Ro - Ri)
    return np.where(
        t < 0.0, 1.0,
        np.where(t >= 1.0, 0.0, np.cos(0.5*np.pi*t)**2)
    )

def y_dimer_basis(d, k, x0):
    return np.exp(-k*(np.asarray(d, float) - x0))

def y_trimer_basis(d, k, x0):
    d = np.maximum(np.asarray(d, float), 1e-12)
    return np.exp(-k*(d - x0))/d

def W2(d, p2):
    d = np.asarray(d, float)
    C = np.asarray(p2["C"], float)
    y = y_dimer_basis(d, p2["k"], p2["x0"])
    acc = y.copy()
    poly = np.zeros_like(d, float)
    for ci in C:
        poly += ci * acc
        acc *= y
    return switch_poly(d, p2["rin"], p2["rout"]) * poly

def deltaW3_raw(d12, d13, d23, p3):
    C = np.asarray(p3["C"], float)
    if C.size != 15:
        raise ValueError("deltaW3 expects 15 coefficients.")
    y0 = y_trimer_basis(d12, p3["k"], p3["x0"])
    y1 = y_trimer_basis(d13, p3["k"], p3["x0"])
    y2 = y_trimer_basis(d23, p3["k"], p3["x0"])

    p0  = y0 + y1 + y2
    p1  = y0*y0 + y1*y1 + y2*y2
    p2m = y0*y1 + y0*y2 + y1*y2
    p3m = y0*y1*y2
    p4  = y0**3 + y1**3 + y2**3
    p5  = (y0*y2*y2 + y0*y1*y1 + y1*y2*y2 + y0*y0*y1 + y1*y1*y2 + y0*y0*y2)
    p6  = y0**4 + y1**4 + y2**4
    p7  = y0*y0*y1*y2 + y0*y1*y2*y2 + y0*y1*y1*y2
    p8  = (y0**3*y2 + y0*y2**3 + y0**3*y1 + y0*y1**3 + y1**3*y2 + y1*y2**3)
    p9  = y0*y0*y2*y2 + y1*y1*y2*y2 + y0*y0*y1*y1
    p10 = y0*y0*y1*y2*y2 + y0*y1*y1*y2*y2 + y0*y0*y1*y1*y2
    p11 = (y1**4*y2 + y0**4*y2 + y0*y2**4 + y0**4*y1 + y1*y2**4 + y0*y1**4)
    p12 = y0**5 + y1**5 + y2**5
    p13 = y0*y1**3*y2 + y0*y1*y2**3 + y0**3*y1*y2
    p14 = (y1**3*y2**2 + y1**2*y2**3 + y0**2*y1**3 +
           y0**2*y2**3 + y0**3*y1**2 + y0**3*y2**2)

    return (C[0]*p0 + C[1]*p1 + C[2]*p2m + C[3]*p3m + C[4]*p4 +
            C[5]*p5 + C[6]*p6 + C[7]*p7 + C[8]*p8 + C[9]*p9 +
            C[10]*p10 + C[11]*p11 + C[12]*p12 + C[13]*p13 + C[14]*p14)

def sf_sum(d12, d13, d23, p3):
    s12 = switch_cos2(d12, p3["Ri"], p3["Ro"])
    s13 = switch_cos2(d13, p3["Ri"], p3["Ro"])
    s23 = switch_cos2(d23, p3["Ri"], p3["Ro"])
    return s12*s13 + s12*s23 + s13*s23

def Umb(d12, d13, d23, p2, p3):
    if ALANINE_MODE:
        return _ala_reference3(d12,d13,d23)
    W2sum = W2(d12, p2) + W2(d13, p2) + W2(d23, p2)
    Vraw  = deltaW3_raw(d12, d13, d23, p3)
    sf    = sf_sum(d12, d13, d23, p3)
    return W2sum + sf*Vraw, sf*Vraw, Vraw

def triangle_mask(d12, d13, d23, eps=1e-12):
    if ALANINE_MODE:
        return np.ones(np.broadcast(d12,d13,d23).shape,dtype=bool)
    d12 = np.asarray(d12, float)
    d13 = np.asarray(d13, float)
    d23 = np.asarray(d23, float)
    return (
        (d12 <= d13 + d23 + eps) &
        (d13 <= d12 + d23 + eps) &
        (d23 <= d12 + d13 + eps) &
        (d12 >= np.abs(d13 - d23) - eps) &
        (d13 >= np.abs(d12 - d23) - eps) &
        (d23 >= np.abs(d12 - d13) - eps)
    )

def W2sum_pairs(d12, d13, d23, p2):
    if ALANINE_MODE:
        return np.zeros(np.broadcast(d12,d13,d23).shape,float)
    return W2(d12, p2) + W2(d13, p2) + W2(d23, p2)

# =============================================================================
# Permutation-invariant feature map (PIP spirit)
#   Input P is always in "physical grid order" (d13, d23, d12).
# =============================================================================
def _features_from_P(P, symmetry_mode="sorted"):
    if ALANINE_MODE:
        return _ala_features(P)
    P = np.asarray(P, float)
    d13 = P[..., 0]
    d23 = P[..., 1]
    d12 = P[..., 2]
    D = np.stack([d12, d13, d23], axis=-1)

    mode = str(symmetry_mode).lower()
    if mode == "none":
        return np.stack([d13, d23, d12], axis=-1)

    if mode == "sorted":
        return np.sort(D, axis=-1)

    s1 = D[..., 0] + D[..., 1] + D[..., 2]
    s2 = D[..., 0]*D[..., 1] + D[..., 0]*D[..., 2] + D[..., 1]*D[..., 2]
    s3 = D[..., 0]*D[..., 1]*D[..., 2]
    return np.stack([s1, s2, s3], axis=-1)

# =============================================================================
# STYLE / FLAGS
#   These are kept matched to the toy plotting script as closely as possible.
# =============================================================================
LAB_X = r"$d_{13}\,(\sigma)$"
LAB_Y = r"$d_{23}\,(\sigma)$"
LAB_Z = r"$d_{12}\,(\sigma)$"
LAB_W3_TRUE = r"$W_3^{\mathrm{True}}\,(k_{\mathrm{B}}T)$"
LAB_W3_PRED = r"$W_3^{\mathrm{Pred}}\,(k_{\mathrm{B}}T)$"
LAB_DW3_TRUE = LAB_W3_TRUE
LAB_DW3_PRED = LAB_W3_PRED
LAB_C = LAB_W3_PRED
LAB_DW3 = LAB_DW3_PRED
LAB_ABSERR = r"$|W_3^{\mathrm{Pred}} - W_3^{\mathrm{True}}|\,(k_{\mathrm{B}}T)$"
ABSERR_CMAP = "magma"

XLIM = (D13_MIN, D13_MAX)
YLIM = (D23_MIN, D23_MAX)
ZLIM = (D12_MIN, D12_MAX)

USE_FIXED_CLIM = False
CLIM_MIN = -30.0
CLIM_MAX = 20.0

fig_scale = 2
FIGSIZE_3D      = (6.0 * fig_scale, 4.5 * fig_scale)
FIGSIZE_RMSE = (4.5 * fig_scale, 4.2 * fig_scale)
FIGSIZE_SLICE2D = (4.8 * fig_scale, 3.9 * fig_scale)
FIGSIZE_PARITY = (4.5 * fig_scale, 4.2 * fig_scale)

AXIS_LABEL_FZ = 36
TICK_LABEL_FZ = 30
CBAR_FZ = AXIS_LABEL_FZ - 5.7
CBAR_TICK_FZ = TICK_LABEL_FZ
LEGEND_FZ = 30
SHOW_TITLES = False

# Keep original figure size; only give tick labels/marks more room.
AXIS_TICK_PAD_3D = 12
AXIS_TICK_PAD_3D_ALL = AXIS_TICK_PAD_3D * 0.5
AXIS_TICK_PAD_2D = 8
CBAR_TICK_PAD = 7
AXIS_TICK_LENGTH = 6.0
AXIS_TICK_WIDTH = 1.25
CBAR_TICK_LENGTH = 5.5
CBAR_TICK_WIDTH = 1.15

VIEW_ELEV = 28
VIEW_AZIM = -140
ORTHOGRAPHIC = True
Z_BOX_SCALE = 0.6

SWAP_XY = False
REVERSE_X = False
REVERSE_Y = False
REVERSE_Z = False
REVERSE_DIAG_X_2D = False
REVERSE_DIAG_Y_2D = False
REVERSE_DIAG_X_3D = False
REVERSE_DIAG_Y_3D = False
REVERSE_FACE_X_2D = False
REVERSE_FACE_Y_2D = False
REVERSE_FACE_X_3D = False
REVERSE_FACE_Y_3D = False

SHOW_CAND_OVERLAY = True
KNN_K = 100
CAND_SHOW_N = 1000
CAND_CMAP = "coolwarm"
CAND_MARKER_SIZE = 28.0
CAND_ALPHA = 0.95
CAND_EDGE_LW = 0.25
CAND_CBAR_PAD = 0.001
CAND_CBAR_YSHIFT = -0.025   # negative moves top colorbar closer to cube
CAND_CBAR_H = 0.03
CAND_CBAR_LABELPAD = 10     # keeps "Candidate score" away from the bar
CAND_CBAR_LABEL = r"Candidate score"

SLICE_N12 = 90
SLICE_N13 = 90
SLICE_VIEW_ELEV = VIEW_ELEV
SLICE_VIEW_AZIM = VIEW_AZIM
FACE_N13 = SLICE_N13
FACE_N23 = SLICE_N13

# GP
KERNEL_SPEC = "m32+rq*se"
ARD = True
INIT_ITERS = 10000
AL_ITERS = 50
AL_BATCH = 5

VAL_FRAC = 0.15
VAL_MIN = 2000
VAL_MAX = 12000

JITTER_DEF      = 5e-4
LIK_NOISE0      = 5e-3
LIK_NOISE_RETRY = 5e-2
WHITE_NUGGET    = 5e-4

PRUNE_NEAR_DUPLICATES = True
PRUNE_EPS = 1e-6

MIN_SEP_MULT = 0.7
ADAPT_MINSEP_DECAY = 0.92
POOL_MIN_FRACTION = 0.01

ACCEPT_TOL = 0.000

STEP0_LSQR_GRAPH_MODE = "grid26"
MAIN_LSQR_GRAPH_MODE  = "grid26"
LSQR_GRAPH_MODE = MAIN_LSQR_GRAPH_MODE

INIT_LSQR_KNN_K = 10
INIT_LSQR_KNN_MAX_DIST = None
LSQR_KNN_K = INIT_LSQR_KNN_K
LSQR_KNN_MAX_DIST = INIT_LSQR_KNN_MAX_DIST

#INIT_EDGE_COUNT       = 12
#INIT_FACE_DIAG_COUNT  = 12
#INIT_SPACE_DIAG_COUNT = 4

INIT_EDGE_COUNT       = int(os.environ.get("INIT_EDGE_COUNT", 1))
INIT_FACE_DIAG_COUNT  = int(os.environ.get("INIT_FACE_DIAG_COUNT", 0))
INIT_SPACE_DIAG_COUNT = int(os.environ.get("INIT_SPACE_DIAG_COUNT", 0))

FIG_DIR = None

# Save the numerical data plotted in each figure without changing the analysis.
SAVE_FIGURE_DATA = True
FIG_DATA_SUBDIR = "figure_data"

MARKER_SCALE = 8.0
MS_TRAIN  = 18 * MARKER_SCALE
MS_VAL    = 18 * MARKER_SCALE
MS_ANCHOR = 80 * MARKER_SCALE
LINE_MST_WIDTH = 3

TRAIN_FACE = "c"
TRAIN_EDGE = "black"
VAL_EDGE = "red"
VAL_FACE = "none"
TRAIN_MARKER = "o"
VAL_MARKER = "^"
MST_COLOR = "black"
MST_Q_COLOR = "tab:orange"

OVERLAY_ORDER_MODE = "mst_last"
Z_VAL, Z_TRAIN, Z_KNN, Z_MST, Z_ANCHOR = 1, 2, 3, 4, 5

FORCE_TEXT2D_ZLABEL = True
ZLABEL_2D_X = -0.11
ZLABEL_2D_Y = 0.50

USE_LATEX = False
if USE_LATEX:
    from matplotlib import rc, rcParams

    path_add = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")
    os.environ["PATH"] = path_add + ":" + os.environ.get("PATH", "")
    os.environ.pop("PERL5LIB", None)

    rc("text", usetex=True)
    rcParams["text.latex.preamble"] = (
        r"\usepackage{amsmath}\usepackage[T1]{fontenc}"
        r"\usepackage{helvet}"
        r"\usepackage{sansmath}\sansmath"
        r"\renewcommand{\familydefault}{\sfdefault}"
        r"\usepackage[strings]{underscore}"
    )
    rc("font", family="sans-serif")
else:
    mpl.rcParams["text.usetex"] = False

# With helvet+sansmath, plain \Delta can render as a black square.
# Use upright sans-serif Delta so it is nonitalic and does not become a box.
LAB_DW3_TRUE = r"$\mathsf{\Delta} W_3^{\mathrm{True}}\,(k_{\mathrm{B}}T)$"
LAB_DW3_PRED = r"$\mathsf{\Delta} W_3^{\mathrm{Pred}}\,(k_{\mathrm{B}}T)$"
LAB_DW3 = LAB_DW3_PRED

# =============================================================================
# Utilities
# =============================================================================
def _safe_output_name(name):
    safe_name = str(name)
    for a, b in (("$", ""), ("\\", ""), ("{", ""), ("}", ""), ("^", ""), (" ", "_"), ("/", "_"), (",", ""), ("(", ""), (")", "")):
        safe_name = safe_name.replace(a, b)
    return safe_name.replace("mathrm", "").replace("W_3", "W3")


def _figure_data_dir():
    global FIG_DIR
    if FIG_DIR is None:
        FIG_DIR = Path("./figs_UNSET")
        FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / FIG_DATA_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_numeric_figure_data(name, values, header, suffix=""):
    if not SAVE_FIGURE_DATA:
        return
    path = _figure_data_dir() / f"{_safe_output_name(name)}{suffix}.csv"
    np.savetxt(
        path,
        np.asarray(values, float),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10e",
    )


def save_scatter_figure_data(name, Pxyz, values):
    Pxyz = np.asarray(Pxyz, float)
    values = np.asarray(values, float).reshape(-1)
    out = np.column_stack([Pxyz[:, 0], Pxyz[:, 1], Pxyz[:, 2], values])
    save_numeric_figure_data(name, out, "d13_sigma,d23_sigma,d12_sigma,plotted_value")


def save_diag_slice_figure_data(name, D12s, D13s, Z, mask):
    out = np.column_stack(
        [
            np.asarray(D13s, float).ravel(order="C"),
            np.asarray(D13s, float).ravel(order="C"),
            np.asarray(D12s, float).ravel(order="C"),
            np.asarray(Z, float).ravel(order="C"),
            np.asarray(mask, bool).ravel(order="C").astype(int),
        ]
    )
    save_numeric_figure_data(name, out, "d13_sigma,d23_sigma,d12_sigma,plotted_value,feasible")


def save_face_slice_figure_data(name, D13f, D23f, Z, mask):
    D13f = np.asarray(D13f, float)
    D23f = np.asarray(D23f, float)
    D12f = np.full_like(D13f, float(D12_MIN))
    out = np.column_stack(
        [
            D13f.ravel(order="C"),
            D23f.ravel(order="C"),
            D12f.ravel(order="C"),
            np.asarray(Z, float).ravel(order="C"),
            np.asarray(mask, bool).ravel(order="C").astype(int),
        ]
    )
    save_numeric_figure_data(name, out, "d13_sigma,d23_sigma,d12_sigma,plotted_value,feasible")


def save_overlay_figure_data(name, info):
    if (not SAVE_FIGURE_DATA) or (not isinstance(info, dict)):
        return

    Pall = np.asarray(info["Pall"], float)
    train_all_idx = np.asarray(info.get("train_all_idx", []), dtype=int)
    train_q_idx = np.asarray(info.get("train_q_idx", []), dtype=int)
    val_idx = np.asarray(info.get("val_idx", []), dtype=int)
    val_idx = val_idx[:min(15000, val_idx.size)]
    anchor_idx = int(info["anchor_idx"])
    knn_idx = np.asarray(info.get("knn_idx", []), dtype=int)
    score_map = info.get("knn_score_map", None) or {}
    step0_edges = info.get("step0_overlay_edges", None) or []
    allowed_all = info.get("mst_allowed_edges_all", None)
    allowed_q = info.get("mst_allowed_edges_q", None)

    base = _safe_output_name(name)
    point_path = _figure_data_dir() / f"{base}_overlay_points.csv"
    with open(point_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "node_index", "d13_sigma", "d23_sigma", "d12_sigma", "candidate_score"])
        for kind, idxs in (("paid", train_all_idx), ("queried_for_mst", train_q_idx), ("validation", val_idx)):
            for g in idxs:
                writer.writerow([kind, int(g), f"{Pall[g, 0]:.10e}", f"{Pall[g, 1]:.10e}", f"{Pall[g, 2]:.10e}", ""])
        if SHOW_CAND_OVERLAY and isinstance(score_map, dict):
            for g in knn_idx:
                score = score_map.get(int(g), np.nan)
                if np.isfinite(score):
                    writer.writerow(["candidate", int(g), f"{Pall[g, 0]:.10e}", f"{Pall[g, 1]:.10e}", f"{Pall[g, 2]:.10e}", f"{float(score):.10e}"])
        writer.writerow(["anchor", anchor_idx, f"{Pall[anchor_idx, 0]:.10e}", f"{Pall[anchor_idx, 1]:.10e}", f"{Pall[anchor_idx, 2]:.10e}", ""])

    def _mst_edges(idxs, allowed_edges):
        idxs = np.asarray(idxs, dtype=int)
        if idxs.size < 2:
            return []
        if allowed_edges is not None and len(allowed_edges) > 0:
            return mst_edges_over_allowed_edges(Pall, idxs, allowed_edges)
        pts = Pall[idxs]
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        G = minimum_spanning_tree(coo_matrix(D)).tocoo()
        return [(int(idxs[i]), int(idxs[j])) for i, j in zip(G.row, G.col)]

    edge_path = _figure_data_dir() / f"{base}_overlay_edges.csv"
    with open(edge_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["kind", "node_index_1", "node_index_2",
             "d13_1_sigma", "d23_1_sigma", "d12_1_sigma",
             "d13_2_sigma", "d23_2_sigma", "d12_2_sigma"]
        )
        for kind, edges in (
            ("mst_paid", _mst_edges(train_all_idx, allowed_all)),
            ("mst_queried", _mst_edges(train_q_idx, allowed_q)),
            ("step0_geometry", step0_edges),
        ):
            for g0, g1 in edges:
                writer.writerow(
                    [kind, int(g0), int(g1),
                     f"{Pall[g0, 0]:.10e}", f"{Pall[g0, 1]:.10e}", f"{Pall[g0, 2]:.10e}",
                     f"{Pall[g1, 0]:.10e}", f"{Pall[g1, 1]:.10e}", f"{Pall[g1, 2]:.10e}"]
                )


def savefig(fig, name, extra_artists=None):
    global FIG_DIR
    if FIG_DIR is None:
        FIG_DIR = Path("./figs_UNSET")
        FIG_DIR.mkdir(exist_ok=True)
    safe_name = _safe_output_name(name)
    path = FIG_DIR / f"{safe_name}.pdf"
    prev_usetex = mpl.rcParams.get("text.usetex", False)
    try:
        fig.canvas.draw()
        fig.savefig(path, bbox_inches="tight", pad_inches=0.40,
                    bbox_extra_artists=extra_artists, dpi=300)
    except Exception as e:
        print(f"[WARN] LaTeX draw failed: {e}\n Falling back to mathtext for this figure only.")
        try:
            mpl.rcParams["text.usetex"] = False
            # Keep labels unchanged in fallback; \mathsf{\Delta} works in both
            # LaTeX and mathtext and stays nonitalic.
            fig.canvas.draw()
        except Exception as e2:
            print(f"[WARN] mathtext draw also failed: {e2}")
        finally:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.40,
                        bbox_extra_artists=extra_artists, dpi=300)
            mpl.rcParams["text.usetex"] = prev_usetex
    plt.show()

def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / (sd + 1e-12)

def y_fit(y):
    m = float(np.mean(y))
    s = float(np.std(y) + 1e-12)
    return m, s

def y2z(y, m, s): return (np.asarray(y, float) - m) / s
def z2y(z, m, s): return np.asarray(z, float) * s + m

def unique_rows_idx(X):
    _, idx = np.unique(np.round(X, 14), axis=0, return_index=True)
    return np.sort(idx)

def rmse(y, yp):
    y = np.ravel(y); yp = np.ravel(yp)
    m = np.isfinite(y) & np.isfinite(yp)
    return float("nan") if (not np.any(m)) else float(np.sqrt(np.mean((y[m]-yp[m])**2)))

def mst_total_length(P, idxs):
    idxs = np.array(sorted(set(map(int, idxs))), dtype=int)
    if idxs.size <= 1:
        return 0.0
    Q = P[idxs]
    delta=Q[:,None,:]-Q[None,:,:]
    if ALANINE_MODE:delta[...,:2]=wrapped(delta[...,:2])
    D = np.linalg.norm(delta,axis=2)
    G = minimum_spanning_tree(coo_matrix(D))
    return float(G.sum())

def _sanitize_training_data(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[m]; y = y[m]
    if X.shape[0] == 0:
        return X, y
    idx = unique_rows_idx(X)
    X = X[idx]; y = y[idx]
    if (not PRUNE_NEAR_DUPLICATES) or X.shape[0] <= 2:
        return X, y
    tree = cKDTree(X)
    keep = np.ones(X.shape[0], bool)
    for i in range(X.shape[0]):
        if not keep[i]:
            continue
        nn = tree.query_ball_point(X[i], r=float(PRUNE_EPS))
        for j in nn:
            if j > i:
                keep[j] = False
    return X[keep], y[keep]

def _norm_from_clim():
    if USE_FIXED_CLIM:
        return mpl.colors.Normalize(vmin=float(CLIM_MIN), vmax=float(CLIM_MAX))
    return None

def _norm_from_masked(Z, mask, p_lo=5.0, p_hi=95.0):
    z = np.asarray(Z, float)
    m = np.asarray(mask, bool) & np.isfinite(z)
    if not np.any(m):
        return None
    v0 = float(np.nanpercentile(z[m], p_lo))
    v1 = float(np.nanpercentile(z[m], p_hi))
    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
        v0 = float(np.nanmin(z[m])); v1 = float(np.nanmax(z[m]))
    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
        return None
    return mpl.colors.Normalize(vmin=v0, vmax=v1)

def _abs_err(pred, truth):
    return np.abs(np.asarray(pred, float) - np.asarray(truth, float))

# =============================================================================
# KERNEL / GP
# =============================================================================
def kernel_from_spec(spec, ard, X=None, y=None):
    spec = spec.replace(" ", "").lower()

    def atom(tok):
        t = tok.lower()
        if t in ("m32","matern32"):
            return gpflow.kernels.Matern32(lengthscales=([1.0]*4 if ALANINE_MODE else [1,1,1]) if ard else 1.0)
        if t in ("m52","matern52"):
            return gpflow.kernels.Matern52(lengthscales=([1.0]*4 if ALANINE_MODE else [1,1,1]) if ard else 1.0)
        if t in ("se","rbf"):
            return gpflow.kernels.SquaredExponential(lengthscales=([1.0]*4 if ALANINE_MODE else [1,1,1]) if ard else 1.0)
        if t in ("rq","ratquad"):
            return gpflow.kernels.RationalQuadratic(lengthscales=([1.0]*4 if ALANINE_MODE else [1,1,1]) if ard else 1.0, alpha=1.0)
        if t in ("lin","linear"):
            return gpflow.kernels.Linear()
        return gpflow.kernels.Matern52(lengthscales=([1.0]*4 if ALANINE_MODE else [1,1,1]) if ard else 1.0)

    if "+" in spec:
        terms = spec.split("+")
        Ks = []
        for term in terms:
            if "*" in term:
                factors = term.split("*")
                k = atom(factors[0])
                for f in factors[1:]:
                    k = k * atom(f)
                Ks.append(k)
            else:
                Ks.append(atom(term))
        K = Ks[0]
        for k2 in Ks[1:]:
            K = K + k2
    elif "*" in spec:
        toks = spec.split("*")
        K = atom(toks[0])
        for t2 in toks[1:]:
            K = K * atom(t2)
    else:
        K = atom(spec)

    if X is not None and y is not None and len(y) > 2:
        var0 = max(float(np.var(y)), 1e-6)

        def init_one(k):
            if hasattr(k, "variance"):
                try: k.variance.assign(var0)
                except Exception: pass
            if hasattr(k, "lengthscales"):
                ls0 = np.clip(np.std(X, axis=0), 0.15, 3.0)
                try: k.lengthscales.assign(ls0 if ard else float(np.mean(ls0)))
                except Exception: pass
            if hasattr(k, "alpha"):
                try: k.alpha.assign(1.0)
                except Exception: pass

        stack = [K]
        while stack:
            kk = stack.pop()
            if hasattr(kk, "kernels"):
                stack.extend(list(kk.kernels))
            else:
                init_one(kk)
    return K

def structured_kernel_1d3d(ard=True):
    k0 = gpflow.kernels.Matern52(active_dims=[0], lengthscales=1.0)
    k1 = gpflow.kernels.Matern52(active_dims=[1], lengthscales=1.0)
    k2 = gpflow.kernels.Matern52(active_dims=[2], lengthscales=1.0)
    if ard:
        k3s = gpflow.kernels.Matern52(active_dims=[0,1,2], lengthscales=[1.0,1.0,1.0])
        k3l = gpflow.kernels.RationalQuadratic(active_dims=[0,1,2],
                                               lengthscales=[1.0,1.0,1.0],
                                               variance=0.5, alpha=1.0)
    else:
        k3s = gpflow.kernels.Matern52(active_dims=[0,1,2], lengthscales=1.0)
        k3l = gpflow.kernels.RationalQuadratic(active_dims=[0,1,2],
                                               lengthscales=1.0,
                                               variance=0.5, alpha=1.0)
    return (k0 + k1 + k2) + (k3s + k3l)

def psd_rescue(model):
    comps = model.kernel.kernels if hasattr(model.kernel, "kernels") else [model.kernel]
    for k in comps:
        if hasattr(k, "lengthscales"):
            ls = np.asarray(k.lengthscales.numpy(), float)
            ls = np.clip(ls, 0.25, 8.0)
            k.lengthscales.assign(ls if ls.ndim else float(ls))
        if hasattr(k, "variance"):
            k.variance.assign(max(float(k.variance.numpy()), 1e-6))
        if hasattr(k, "alpha"):
            k.alpha.assign(np.clip(float(k.alpha.numpy()), 0.2, 5.0))
    model.likelihood.variance.assign(max(float(model.likelihood.variance.numpy()), 1e-2))

def _try_minimize(m, iters):
    gpflow.optimizers.Scipy().minimize(
        m.training_loss, m.trainable_variables,
        options={"maxiter": int(iters), "disp": False}
    )

def warm_start_params(m_new, m_prev):
    if m_prev is None:
        return
    try:
        pd_prev = gpf_util.parameter_dict(m_prev)
        pd_new  = gpf_util.parameter_dict(m_new)
    except Exception:
        return
    for name, v_new in pd_new.items():
        v_old = pd_prev.get(name, None)
        if v_old is None:
            continue
        try:
            if v_new.shape == v_old.shape:
                v_new.assign(v_old)
        except Exception:
            pass

def build_gpr(X, z, kernel_mode, spec, ard, iters, m_prev=None):
    X = np.asarray(X, float)
    z = np.asarray(z, float).reshape(-1)
    if X.shape[0] < 2:
        raise RuntimeError("Not enough training points.")

    retry_jitters = [JITTER_DEF, 5e-4, 2e-3, 1e-2]
    retry_noises  = [LIK_NOISE0, LIK_NOISE_RETRY, 5e-2, 1e-1]
    last_err = None

    for jj, nv in zip(retry_jitters, retry_noises):
        try:
            gpflow.config.set_default_jitter(float(jj))
            Xtf = tf.convert_to_tensor(X, tf.float64)
            ztf = tf.convert_to_tensor(z.reshape(-1, 1), tf.float64)

            if str(kernel_mode).lower() == "structured":
                base = structured_kernel_1d3d(ard=ard)
            else:
                base = kernel_from_spec(spec, ard, X, z)

            white = gpflow.kernels.White(variance=WHITE_NUGGET)
            gpflow.set_trainable(white.variance, False)
            K = base + white

            m = gpflow.models.GPR((Xtf, ztf), kernel=K,
                                  mean_function=gpflow.mean_functions.Zero())
            warm_start_params(m, m_prev)
            m.likelihood.variance.assign(float(nv))
            psd_rescue(m)

            try:
                _try_minimize(m, iters)
            except Exception:
                psd_rescue(m)
                _try_minimize(m, max(200, iters // 2))

            psd_rescue(m)
            mu, var = m.predict_f(Xtf)
            mu = mu.numpy(); var = var.numpy()
            if (not np.all(np.isfinite(mu))) or (not np.all(np.isfinite(var))) or np.any(var <= 0):
                raise FloatingPointError("predict_f produced non-finite/invalid variance.")
            return m
        except Exception as e:
            last_err = e

    raise RuntimeError(f"GP build failed after retries. Last error: {last_err}")

def predict_mu_std_z(m, Xnew):
    Xtf = tf.convert_to_tensor(np.asarray(Xnew, float), tf.float64)
    try:
        mu, var = m.predict_f(Xtf)
    except Exception:
        psd_rescue(m)
        mu, var = m.predict_f(Xtf)
    mu = mu.numpy().ravel()
    var = np.nan_to_num(var.numpy().ravel(), nan=1e-6, posinf=1e6, neginf=1e-12)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mu, std

# =============================================================================
# Plotting helpers
# =============================================================================
def _apply_axis_mapping(Pxyz, swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    return Pxyz[:, [1, 0, 2]] if use_swap else Pxyz

def _mapped_labels(swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    return (LAB_Y, LAB_X, LAB_Z) if use_swap else (LAB_X, LAB_Y, LAB_Z)

def _mapped_limits(swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    return (YLIM, XLIM, ZLIM) if use_swap else (XLIM, YLIM, ZLIM)

def _apply_axis_inversions(ax, swap_xy=None):
    use_swap = SWAP_XY if swap_xy is None else bool(swap_xy)
    if REVERSE_X:
        ax.invert_yaxis() if use_swap else ax.invert_xaxis()
    if REVERSE_Y:
        ax.invert_xaxis() if use_swap else ax.invert_yaxis()
    if REVERSE_Z:
        ax.invert_zaxis()

def _compact_number_label(x, pos=None):
    """Compact tick labels: -10 instead of -10.0, but keep decimals if needed."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) < 1e-12:
        x = 0.0
    if np.isclose(x, np.round(x), atol=1e-9):
        s = f"{int(np.round(x))}"
    else:
        s = f"{x:g}"
    if mpl.rcParams.get("axes.unicode_minus", True):
        s = s.replace("-", "−")
    return s

COMPACT_TICK_FORMATTER = FuncFormatter(_compact_number_label)


def _distance_ticks(lim):
    lo = int(np.ceil(float(lim[0])))
    hi = int(np.floor(float(lim[1])))
    return np.arange(lo, hi + 1, 1.0) if hi >= lo else None

def _apply_distance_ticks_3d(ax, xlim, ylim, zlim=None):
    xt = _distance_ticks(xlim)
    yt = _distance_ticks(ylim)
    if xt is not None: ax.set_xticks(xt)
    if yt is not None: ax.set_yticks(yt)
    if zlim is not None:
        zt = _distance_ticks(zlim)
        if zt is not None: ax.set_zticks(zt)
    ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

def _format_energy_zaxis(ax):
    # Prefer integer energy ticks where possible. If the z range is too small,
    # MaxNLocator will still allow decimals, but the formatter removes .0.
    try:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, integer=True))
    except TypeError:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.zaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

def _format_colorbar_ticks(cb):
    cb.ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    cb.ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

def _adjust_pinned_zlabel_for_ticks(ax, z_art):
    if z_art is None:
        return
    labels = [_compact_number_label(v) for v in ax.get_zticks()]
    max_chars = max((len(s) for s in labels), default=0)
    # Shift only when tick labels are genuinely long, so short-label plots do
    # not get an unnecessarily large gap.
    x = float(ZLABEL_2D_X) - max(0, max_chars - 3) * 0.014
    z_art.set_position((x, float(ZLABEL_2D_Y)))

def _apply_distance_ticks_2d(ax, xlim, ylim):
    xt = _distance_ticks(xlim)
    yt = _distance_ticks(ylim)
    if xt is not None: ax.set_xticks(xt)
    if yt is not None: ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(COMPACT_TICK_FORMATTER)
    ax.yaxis.set_major_formatter(COMPACT_TICK_FORMATTER)

def _make_3d_ticks_visible(ax):
    # mplot3d tick marks can become visually hidden unless their 3D tick
    # factors are explicitly made outward-facing.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["tick"]["inward_factor"] = 0.0
            axis._axinfo["tick"]["outward_factor"] = 0.28
            axis._axinfo["tick"]["linewidth"] = {
                True: AXIS_TICK_WIDTH,
                False: AXIS_TICK_WIDTH,
            }
        except Exception:
            pass

def _add_top_colorbar(fig, ax, mappable, label):
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y1 + CAND_CBAR_PAD + CAND_CBAR_YSHIFT, pos.width, CAND_CBAR_H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(label, fontsize=CBAR_FZ, labelpad=CAND_CBAR_LABELPAD)
    cb.ax.tick_params(labelsize=TICK_LABEL_FZ - 6, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)
    return cb

def _pin_3d_zlabel_2d(ax, label, fontsize, x=ZLABEL_2D_X, y=ZLABEL_2D_Y):
    try:
        ax.set_zlabel("")
    except Exception:
        pass
    try:
        art = ax.text2D(
            float(x), float(y), label,
            transform=ax.transAxes,
            fontsize=float(fontsize),
            rotation=90,
            va="center", ha="right",
            clip_on=False
        )
        return art
    except Exception:
        return None

def plot_scatter3d_color(Pxyz, C, clabel, name, title,
                         overlay=None, cmap="viridis", norm=None, cbar_ticks=None,
                         scatter_swap_xy=False):
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    try: ax.computed_zorder = False
    except Exception: pass
    if ORTHOGRAPHIC:
        try: ax.set_proj_type("ortho")
        except Exception: pass

    Q = _apply_axis_mapping(Pxyz, swap_xy=scatter_swap_xy)
    lx, ly, lz = _mapped_labels(swap_xy=scatter_swap_xy)
    xlim, ylim, zlim = _mapped_limits(swap_xy=scatter_swap_xy)

    sc = ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2],
                    c=C, s=8, cmap=cmap, norm=norm,
                    linewidths=0, alpha=0.85,
                    depthshade=False, zorder=0)

    ax.set_xlabel(lx, fontsize=AXIS_LABEL_FZ, labelpad=26)
    ax.set_ylabel(ly, fontsize=AXIS_LABEL_FZ, labelpad=26)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, lz, AXIS_LABEL_FZ, x=ZLABEL_2D_X, y=ZLABEL_2D_Y)
    else:
        ax.set_zlabel(lz, fontsize=AXIS_LABEL_FZ, labelpad=10)

    if SHOW_TITLES:
        ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=0)

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    _apply_distance_ticks_3d(ax, xlim, ylim, zlim)
    try:
        ax.set_box_aspect(((xlim[1]-xlim[0]), (ylim[1]-ylim[0]), Z_BOX_SCALE*(zlim[1]-zlim[0])))
    except Exception:
        pass

    ax.view_init(elev=float(VIEW_ELEV), azim=float(VIEW_AZIM))
    _apply_axis_inversions(ax, swap_xy=scatter_swap_xy)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    _adjust_pinned_zlabel_for_ticks(ax, z_art)
    _make_3d_ticks_visible(ax)

    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=TICK_LABEL_FZ - 6, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)

    extra = overlay(ax) if callable(overlay) else None
    if isinstance(extra, dict) and extra.get("knn_mappable", None) is not None:
        _add_top_colorbar(fig, ax, extra["knn_mappable"], extra.get("knn_label", CAND_CBAR_LABEL))

    extra_artists = [z_art] if z_art is not None else None
    save_scatter_figure_data(name, Pxyz, C)
    if callable(overlay) and hasattr(overlay, "figure_data"):
        save_overlay_figure_data(name, overlay.figure_data)
    savefig(fig, name, extra_artists=extra_artists)

# =============================================================================
# Slice grids + plotting
# =============================================================================
def build_diag_slice_grid():
    if ALANINE_MODE:
        return _ala_diag_grid()
    d12s = np.linspace(D12_MIN, D12_MAX, SLICE_N12)
    d13s = np.linspace(D13_MIN, D13_MAX, SLICE_N13)
    D12s, D13s = np.meshgrid(d12s, d13s, indexing="ij")
    D23s = D13s.copy()
    m = triangle_mask(D12s, D13s, D23s)
    return d12s, d13s, D12s, D13s, D23s, m

def build_face_slice_grid():
    if ALANINE_MODE:
        return _ala_face_grid()
    d13f = np.linspace(D13_MIN, D13_MAX, FACE_N13)
    d23f = np.linspace(D23_MIN, D23_MAX, FACE_N23)
    D23f, D13f = np.meshgrid(d23f, d13f, indexing="ij")
    D12f = np.full_like(D13f, float(D12_MIN))
    m = triangle_mask(D12f, D13f, D23f)
    return d13f, d23f, D13f, D23f, D12f, m

def plot_slice_2d_heat(D12s, D13s, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    x = D13s[0, :]
    y = D12s[:, 0]
    fig, ax = plt.subplots(figsize=FIGSIZE_SLICE2D)
    pc = ax.pcolormesh(x, y, Zm, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(r"$d_{13}=d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ + 6, labelpad=10)
    ax.set_ylabel(r"$d_{12}\,(\sigma)$", fontsize=AXIS_LABEL_FZ + 6, labelpad=10)

    _apply_distance_ticks_2d(ax, (x.min(), x.max()), (y.min(), y.max()))

    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=AXIS_TICK_PAD_2D,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    if REVERSE_DIAG_X_2D: ax.invert_xaxis()
    if REVERSE_DIAG_Y_2D: ax.invert_yaxis()
    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)
    save_diag_slice_figure_data(name, D12s, D13s, Z, m)
    savefig(fig, name)

def plot_slice_3d_surface(D12s, D13s, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    ax.tick_params(axis="z", pad=50)
    for lab in ax.zaxis.get_ticklabels():
        lab.set_horizontalalignment("right")
        lab.set_verticalalignment("center")
        lab.set_rotation(0)
    try: ax.computed_zorder = False
    except Exception: pass
    if ORTHOGRAPHIC:
        try: ax.set_proj_type("ortho")
        except Exception: pass
    surf = ax.plot_surface(D13s, D12s, Zm, cmap=cmap, norm=norm, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$d_{13}=d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=26)
    ax.set_ylabel(r"$d_{12}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=26)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, clabel, AXIS_LABEL_FZ)
    else:
        ax.set_zlabel(clabel, fontsize=AXIS_LABEL_FZ, labelpad=16)

    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    ax.set_xlim(D13_MIN, D13_MAX); ax.set_ylim(D12_MIN, D12_MAX)
    _apply_distance_ticks_3d(ax, (D13_MIN, D13_MAX), (D12_MIN, D12_MAX), None)
    if REVERSE_DIAG_X_3D: ax.invert_xaxis()
    if REVERSE_DIAG_Y_3D: ax.invert_yaxis()
    try:
        ax.set_box_aspect(((D13_MAX - D13_MIN), (D12_MAX - D12_MIN), Z_BOX_SCALE * (D12_MAX - D12_MIN)))
    except Exception:
        pass
    ax.view_init(elev=float(SLICE_VIEW_ELEV), azim=float(SLICE_VIEW_AZIM))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    _format_energy_zaxis(ax)
    _adjust_pinned_zlabel_for_ticks(ax, z_art)
    _make_3d_ticks_visible(ax)
    cb = fig.colorbar(surf, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=TICK_LABEL_FZ - 6, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)
    extra_artists = [z_art] if z_art is not None else None
    save_diag_slice_figure_data(name, D12s, D13s, Z, m)
    savefig(fig, name, extra_artists=extra_artists)

def plot_face_2d_heat(d13f, d23f, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    fig, ax = plt.subplots(figsize=FIGSIZE_SLICE2D)
    pc = ax.pcolormesh(d13f, d23f, Zm, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlabel(r"$d_{13}\,(\sigma)$", fontsize=AXIS_LABEL_FZ + 6, labelpad=10)
    ax.set_ylabel(r"$d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ + 6, labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=AXIS_TICK_PAD_2D,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    if REVERSE_FACE_X_2D: ax.invert_xaxis()
    if REVERSE_FACE_Y_2D: ax.invert_yaxis()
    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=CBAR_TICK_FZ, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)
    _apply_distance_ticks_2d(ax, (d13f.min(), d13f.max()), (d23f.min(), d23f.max()))
    D13out, D23out = np.meshgrid(d13f, d23f, indexing="xy")
    save_face_slice_figure_data(name, D13out, D23out, Z, m)
    savefig(fig, name)

def plot_face_3d_surface(D13f, D23f, Z, m, name, title, clabel, cmap="viridis", norm=None):
    Zm = np.where(m, Z, np.nan)
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    ax.tick_params(axis="z", pad=50)
    for lab in ax.zaxis.get_ticklabels():
        lab.set_horizontalalignment("right")
        lab.set_verticalalignment("center")
        lab.set_rotation(0)
    try: ax.computed_zorder = False
    except Exception: pass
    if ORTHOGRAPHIC:
        try: ax.set_proj_type("ortho")
        except Exception: pass
    surf = ax.plot_surface(D13f, D23f, Zm, cmap=cmap, norm=norm, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$d_{13}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=26)
    ax.set_ylabel(r"$d_{23}\,(\sigma)$", fontsize=AXIS_LABEL_FZ, labelpad=26)

    z_art = None
    if FORCE_TEXT2D_ZLABEL:
        z_art = _pin_3d_zlabel_2d(ax, clabel, AXIS_LABEL_FZ)
    else:
        ax.set_zlabel(clabel, fontsize=AXIS_LABEL_FZ, labelpad=16)

    if SHOW_TITLES: ax.set_title(title, fontsize=LEGEND_FZ + 2, pad=10)
    ax.set_xlim(D13_MIN, D13_MAX); ax.set_ylim(D23_MIN, D23_MAX)
    _apply_distance_ticks_3d(ax, (D13_MIN, D13_MAX), (D23_MIN, D23_MAX), None)
    if REVERSE_FACE_X_3D: ax.invert_xaxis()
    if REVERSE_FACE_Y_3D: ax.invert_yaxis()
    try:
        ax.set_box_aspect(((D13_MAX - D13_MIN), (D23_MAX - D23_MIN), Z_BOX_SCALE * (D23_MAX - D23_MIN)))
    except Exception:
        pass
    ax.view_init(elev=float(SLICE_VIEW_ELEV), azim=float(SLICE_VIEW_AZIM))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.tick_params(axis="z", labelsize=TICK_LABEL_FZ - 6, pad=AXIS_TICK_PAD_3D_ALL,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    _format_energy_zaxis(ax)
    _adjust_pinned_zlabel_for_ticks(ax, z_art)
    _make_3d_ticks_visible(ax)
    cb = fig.colorbar(surf, ax=ax, shrink=0.78, pad=0.05)
    cb.set_label(clabel, fontsize=AXIS_LABEL_FZ, labelpad=12)
    cb.ax.tick_params(labelsize=TICK_LABEL_FZ - 6, pad=CBAR_TICK_PAD,
                      length=CBAR_TICK_LENGTH, width=CBAR_TICK_WIDTH)
    _format_colorbar_ticks(cb)
    extra_artists = [z_art] if z_art is not None else None
    save_face_slice_figure_data(name, D13f, D23f, Z, m)
    savefig(fig, name, extra_artists=extra_artists)

# =============================================================================
# GP predict on slice/face (uses symmetry_mode feature map)
# =============================================================================
def gp_predict_on_slice(model, xmu, xsd, ymu, ysd, D12s, D13s, symmetry_mode):
    P = np.c_[D13s.ravel(), D13s.ravel(), D12s.ravel()]
    Pf = _features_from_P(P, symmetry_mode=symmetry_mode)
    Xs = standardize_apply(Pf, xmu, xsd)
    mu_z, std_z = predict_mu_std_z(model, Xs)
    mu_y = z2y(mu_z, ymu, ysd)
    std_y = std_z * ysd
    return mu_y.reshape(D12s.shape), std_y.reshape(D12s.shape)

def gp_predict_on_face(model, xmu, xsd, ymu, ysd, D13f, D23f, D12f, symmetry_mode):
    P = np.c_[D13f.ravel(), D23f.ravel(), D12f.ravel()]
    Pf = _features_from_P(P, symmetry_mode=symmetry_mode)
    Xs = standardize_apply(Pf, xmu, xsd)
    mu_z, std_z = predict_mu_std_z(model, Xs)
    mu_y = z2y(mu_z, ymu, ysd)
    std_y = std_z * ysd
    return mu_y.reshape(D13f.shape), std_y.reshape(D13f.shape)

def oracle_on_slice(D12s, D13s, p2, p3):
    if ALANINE_MODE:
        return _ala_reference3(np.zeros_like(D13s),D13s,D12s)
    D23s = D13s
    U, W3sw, W3raw = Umb(D12s, D13s, D23s, p2, p3)
    return U, W3sw, W3raw

def oracle_on_face(D13f, D23f, D12f, p2, p3):
    U, W3sw, W3raw = Umb(D12f, D13f, D23f, p2, p3)
    return U, W3sw, W3raw

# =============================================================================
# Grid indexing / neighbors
# =============================================================================
def ijk_to_g(i12, i13, i23, n12, n13, n23):
    return (int(i12) * n13 + int(i13)) * n23 + int(i23)

def g_to_ijk(g, n12, n13, n23):
    g = int(g)
    i12, rem = divmod(g, n13 * n23)
    i13, i23 = divmod(rem, n23)
    return int(i12), int(i13), int(i23)

def neighbors_26(g, n12, n13, n23):
    if ALANINE_MODE:
        return _ala_neighbors(g,n12,n13,n23)
    i12, i13, i23 = g_to_ijk(g, n12, n13, n23)
    out = []
    for di12 in (-1, 0, 1):
        for di13 in (-1, 0, 1):
            for di23 in (-1, 0, 1):
                if di12 == 0 and di13 == 0 and di23 == 0:
                    continue
                j12 = i12 + di12
                j13 = i13 + di13
                j23 = i23 + di23
                if 0 <= j12 < n12 and 0 <= j13 < n13 and 0 <= j23 < n23:
                    out.append(ijk_to_g(j12, j13, j23, n12, n13, n23))
    return out

# =============================================================================
# Strict connectivity (path paying)
# =============================================================================
def manhattan_walk_add_3d(train_set, start_g, target_g, n12, n13, n23, allowed_mask_flat):
    start_g = int(start_g); target_g = int(target_g)
    if (not allowed_mask_flat[start_g]) or (not allowed_mask_flat[target_g]):
        return []
    i12, i13, i23 = g_to_ijk(start_g, n12, n13, n23)
    j12, j13, j23 = g_to_ijk(target_g, n12, n13, n23)

    added = []
    train_set.add(start_g); added.append(start_g)

    if i12 != j12:
        step = 1 if j12 > i12 else -1
        while i12 != j12:
            i12 += step
            g = ijk_to_g(i12, i13, i23, n12, n13, n23)
            if allowed_mask_flat[g]:
                train_set.add(g); added.append(g)

    if i13 != j13:
        step = 1 if j13 > i13 else -1
        while i13 != j13:
            i13 += step
            g = ijk_to_g(i12, i13, i23, n12, n13, n23)
            if allowed_mask_flat[g]:
                train_set.add(g); added.append(g)

    if i23 != j23:
        step = 1 if j23 > i23 else -1
        while i23 != j23:
            i23 += step
            g = ijk_to_g(i12, i13, i23, n12, n13, n23)
            if allowed_mask_flat[g]:
                train_set.add(g); added.append(g)
    return added

def shortest_path_add_3d(train_set, start_g, target_g, P_full, n12, n13, n23, allowed_mask_flat):
    start_g = int(start_g); target_g = int(target_g)
    if start_g == target_g:
        if allowed_mask_flat[start_g]:
            train_set.add(start_g); return [start_g]
        return []
    if (not allowed_mask_flat[start_g]) or (not allowed_mask_flat[target_g]):
        return []

    train_set.add(start_g)
    added = [start_g]
    dist = {start_g: 0.0}
    prev = {start_g: None}
    pq = [(0.0, start_g)]

    while pq:
        dcur, g = heapq.heappop(pq)
        if g == target_g:
            break
        if dcur != dist.get(g, np.inf):
            continue
        for h in neighbors_26(g, n12, n13, n23):
            h = int(h)
            if not allowed_mask_flat[h]:
                continue
            step = float(np.linalg.norm(_ala_displacement(P_full[h],P_full[g]) if ALANINE_MODE else P_full[h]-P_full[g]))
            nd = dcur + step
            if nd < dist.get(h, np.inf):
                dist[h] = nd
                prev[h] = g
                heapq.heappush(pq, (nd, h))

    if target_g not in prev:
        return manhattan_walk_add_3d(train_set, start_g, target_g, n12, n13, n23, allowed_mask_flat)

    path = []
    g = target_g
    while g is not None:
        path.append(int(g))
        g = prev[g]
    path.reverse()

    for g in path[1:]:
        if allowed_mask_flat[int(g)]:
            train_set.add(int(g)); added.append(int(g))
    return added

# =============================================================================
# LSQR force integration helpers
# =============================================================================
def build_edges_from_nodes(nodes_set, n12, n13, n23, feas_mask_flat):
    nodes = np.array(sorted(set(map(int, nodes_set))), dtype=int)
    node_set = set(nodes.tolist())
    edges = set()
    for g in nodes:
        for h in neighbors_26(g, n12, n13, n23):
            if (h in node_set) and feas_mask_flat[h]:
                i, j = (g, h) if g < h else (h, g)
                edges.add((i, j))
    return list(edges)

def build_edges_axis1_from_nodes(nodes_set, n12, n13, n23, feas_mask_flat):
    nodes = np.array(sorted(set(map(int, nodes_set))), dtype=int)
    node_set = set(nodes.tolist())
    edges = set()
    for g in nodes:
        g = int(g)
        if not feas_mask_flat[g]:
            continue
        i12, i13, i23 = g_to_ijk(g, n12, n13, n23)
        for di12, di13, di23 in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            j12, j13, j23 = i12 + di12, i13 + di13, i23 + di23
            if 0 <= j12 < n12 and 0 <= j13 < n13 and 0 <= j23 < n23:
                h = ijk_to_g(j12, j13, j23, n12, n13, n23)
                if (h in node_set) and feas_mask_flat[h]:
                    a, b = (g, h) if g < h else (h, g)
                    edges.add((a, b))
    return list(edges)

def build_edges_knn_from_nodes(nodes_set, P_full, k=10, max_dist=None):
    nodes = np.array(sorted(set(map(int, nodes_set))), dtype=int)
    if nodes.size < 2:
        return []
    pts = P_full[nodes]
    tree = cKDTree(pts)
    k_use = int(min(max(1, k), nodes.size - 1))
    dists, idxs = tree.query(pts, k=k_use + 1)
    dists = np.atleast_2d(dists)
    idxs  = np.atleast_2d(idxs)

    edges = set()
    for a in range(nodes.size):
        ga = int(nodes[a])
        for t in range(1, k_use + 1):
            b = int(idxs[a, t])
            if b < 0 or b >= nodes.size:
                continue
            if (max_dist is not None) and (float(dists[a, t]) > float(max_dist)):
                continue
            gb = int(nodes[b])
            i, j = (ga, gb) if ga < gb else (gb, ga)
            if i != j:
                edges.add((i, j))
    return list(edges)

def lsqr_U_on_known_nodes_3d(known_nodes, anchor_g, P_full, Fx, Fy, Fz,
                             n12, n13, n23, feas_mask_flat,
                             atol=1e-10, btol=1e-10, iter_lim=8000,
                             graph_mode=None, knn_k=None, knn_max_dist=None,
                             extra_edges=None,
                             integration_mode=None, p2=None, p3=None,
                             quad_order=None, fd_eps=None, edge_batch_size=None):
    known_nodes = np.array(sorted(set(map(int, known_nodes))), dtype=int)
    nk = known_nodes.size
    if nk == 0:
        return np.array([], float), known_nodes

    loc = {int(g): i for i, g in enumerate(known_nodes)}
    if int(anchor_g) not in loc:
        return np.full(nk, np.nan, float), known_nodes

    mode = (LSQR_GRAPH_MODE if graph_mode is None else str(graph_mode)).lower()

    if mode == "knn":
        k_use = int(knn_k if knn_k is not None else LSQR_KNN_K)
        r_use = knn_max_dist if knn_max_dist is not None else LSQR_KNN_MAX_DIST
        edges = build_edges_knn_from_nodes(known_nodes, P_full, k=k_use, max_dist=r_use)
    elif mode == "axis1":
        edges = build_edges_axis1_from_nodes(known_nodes, n12, n13, n23, feas_mask_flat)
    else:
        edges = build_edges_from_nodes(known_nodes, n12, n13, n23, feas_mask_flat)

    if extra_edges:
        es = set((int(a), int(b)) if int(a) < int(b) else (int(b), int(a)) for (a, b) in edges)
        for (a, b) in extra_edges:
            a = int(a); b = int(b)
            if a == b:
                continue
            if (a not in loc) or (b not in loc):
                continue
            if (not feas_mask_flat[a]) or (not feas_mask_flat[b]):
                continue
            i, j = (a, b) if a < b else (b, a)
            es.add((i, j))
        edges = list(es)

    rows, cols, data, b = [], [], [], []
    n_skip_nonfinite = 0
    n_skip_clip = 0

    int_mode = (LSQR_INTEGRATION_MODE if integration_mode is None else str(integration_mode)).lower()
    use_continuous = int_mode in ("continuous", "cont", "quad", "quadrature")
    q_order = int(LSQR_EDGE_QUAD_ORDER if quad_order is None else quad_order)
    eps_fd = float(LSQR_FD_EPS if fd_eps is None else fd_eps)
    batch_size = int(LSQR_EDGE_BATCH_SIZE if edge_batch_size is None else edge_batch_size)

    edge_pairs = []
    for (ii, jj) in edges:
        ii = int(ii); jj = int(jj)
        if (ii not in loc) or (jj not in loc):
            continue
        edge_pairs.append((ii, jj))

    if use_continuous:
        if (p2 is None) or (p3 is None):
            raise ValueError("Continuous LSQR requires p2 and p3 so the oracle gradient can be evaluated along each edge.")
        dws = _edge_work_quadrature_3d_batch(
            P_full, edge_pairs, p2, p3,
            quad_order=q_order,
            fd_eps=eps_fd,
            edge_batch_size=batch_size,
        )
    else:
        dws = np.full(len(edge_pairs), np.nan, float)
        for e, (ii, jj) in enumerate(edge_pairs):
            Fx_i, Fy_i, Fz_i = Fx[ii], Fy[ii], Fz[ii]
            Fx_j, Fy_j, Fz_j = Fx[jj], Fy[jj], Fz[jj]
            if not (np.isfinite(Fx_i) and np.isfinite(Fy_i) and np.isfinite(Fz_i) and
                    np.isfinite(Fx_j) and np.isfinite(Fy_j) and np.isfinite(Fz_j)):
                continue

            Fx_mid = 0.5 * (Fx_i + Fx_j)
            Fy_mid = 0.5 * (Fy_i + Fy_j)
            Fz_mid = 0.5 * (Fz_i + Fz_j)

            dx = P_full[jj] - P_full[ii]
            dws[e] = +(Fx_mid * dx[0] + Fy_mid * dx[1] + Fz_mid * dx[2])

    for (ii, jj), dw in zip(edge_pairs, dws):
        ii = int(ii); jj = int(jj)
        if not np.isfinite(dw):
            n_skip_nonfinite += 1
            continue
        if (DW_CLIP is not None) and (abs(float(dw)) > float(DW_CLIP)):
            n_skip_clip += 1
            continue

        r = len(b)
        b.append(float(dw))
        rows.extend([r, r]); cols.extend([loc[ii], loc[jj]]); data.extend([-1.0, +1.0])

        r2 = len(b)
        w = np.sqrt(1e-3)
        b.append(0.0)
        rows.extend([r2, r2]); cols.extend([loc[ii], loc[jj]]); data.extend([-w, +w])

    M = len(b)
    if LSQR_PRINT_EQUATION_STATS:
        print(f"[LSQR build] mode={mode} integration={int_mode} nk={nk} edges_in={len(edges)} "
              f"edges_used={len(edge_pairs)} M={M} skip_nonfinite={n_skip_nonfinite} "
              f"skip_clip={n_skip_clip} quad={q_order if use_continuous else 'endpoint'}")

    if M == 0:
        return np.full(nk, np.nan, float), known_nodes

    Asub = coo_matrix((np.asarray(data, float),
                       (np.asarray(rows, int), np.asarray(cols, int))),
                      shape=(M, nk)).tocsr()
    b = np.asarray(b, float)

    a_loc = loc[int(anchor_g)]
    keep = np.arange(nk, dtype=int) != a_loc

    U = np.zeros(nk, float)
    if np.any(keep):
        sol = lsqr(Asub[:, keep], b, atol=atol, btol=btol, iter_lim=iter_lim)[0]
        U[keep] = sol
    U[a_loc] = 0.0
    U -= U[a_loc]
    return U, known_nodes



def _Umb_value_from_P_3d(P, p2, p3):
    """Return Umb in physical P-space, where P=(d13,d23,d12)."""
    P = np.asarray(P, float)
    d13 = P[..., 0]
    d23 = P[..., 1]
    d12 = P[..., 2]
    U, _, _ = Umb(d12, d13, d23, p2, p3)
    return np.asarray(U, float)


def _grad_Umb_fd_3d(P, p2, p3, fd_eps=1e-4):
    """Central finite-difference gradient of Umb in P=(d13,d23,d12) space."""
    P = np.asarray(P, float)
    scalar_input = (P.ndim == 1)
    Q = P.reshape(1, 3) if scalar_input else P.reshape(-1, 3)

    eps = float(fd_eps)
    G = np.empty_like(Q, dtype=float)
    for ax in range(3):
        step = np.zeros(3, float)
        step[ax] = eps
        up = _Umb_value_from_P_3d(Q + step[None, :], p2, p3)
        um = _Umb_value_from_P_3d(Q - step[None, :], p2, p3)
        G[:, ax] = (np.asarray(up, float).ravel() - np.asarray(um, float).ravel()) / (2.0 * eps)

    if scalar_input:
        return G[0, 0], G[0, 1], G[0, 2]
    return G[:, 0], G[:, 1], G[:, 2]


def _edge_work_quadrature_3d(p0, p1, p2, p3, quad_order=16, fd_eps=1e-4):
    """Integrate dU = grad(U).dx from p0 to p1 in 3D P-space.

    Coordinates are P=(d13,d23,d12), so the returned gradient components are
    (dU/dd13, dU/dd23, dU/dd12).  This mirrors the 2D sibling's
    _edge_work_quadrature_2d(...) but uses finite-difference gradients of the
    smooth MD/PIP oracle Umb(...) because this script does not have analytic
    force expressions.
    """
    if ALANINE_MODE:
        return _ALA.edge_work(p0,p1)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    dx = p1 - p0
    if not np.all(np.isfinite(dx)):
        return np.nan
    if float(np.linalg.norm(dx)) == 0.0:
        return 0.0

    qx, qw = np.polynomial.legendre.leggauss(int(quad_order))
    t = 0.5 * (qx + 1.0)
    w = 0.5 * qw
    pts = p0[None, :] + t[:, None] * dx[None, :]

    gx, gy, gz = _grad_Umb_fd_3d(pts, p2, p3, fd_eps=fd_eps)
    integrand = (
        np.asarray(gx, float) * dx[0]
        + np.asarray(gy, float) * dx[1]
        + np.asarray(gz, float) * dx[2]
    )
    if not np.all(np.isfinite(integrand)):
        return np.nan
    return float(np.sum(w * integrand))




def _edge_work_quadrature_3d_batch(P_full, edge_pairs, p2, p3, quad_order=16, fd_eps=1e-4, edge_batch_size=3000):
    """Vectorized continuous edge integration for LSQR.

    Computes dU_ij = int_0^1 grad(U)[p_i + t(p_j-p_i)] . (p_j-p_i) dt
    for many graph edges.  P-space is P=(d13,d23,d12), so the gradient
    components are (dU/dd13, dU/dd23, dU/dd12).
    """
    edge_pairs = np.asarray(edge_pairs, dtype=int)
    if edge_pairs.size == 0:
        return np.array([], float)

    edge_pairs = edge_pairs.reshape(-1, 2)
    n_edges = edge_pairs.shape[0]
    dU_edges = np.full(n_edges, np.nan, float)

    qn = int(max(2, quad_order))
    xg, wg = np.polynomial.legendre.leggauss(qn)
    ts = 0.5 * (xg + 1.0)
    wts = 0.5 * wg

    bs = int(max(1, edge_batch_size))
    for a0 in range(0, n_edges, bs):
        a1 = min(a0 + bs, n_edges)
        eb = edge_pairs[a0:a1]
        p0 = np.asarray(P_full[eb[:, 0]], float)
        p1 = np.asarray(P_full[eb[:, 1]], float)
        dx = p1 - p0
        lengths = np.linalg.norm(dx, axis=1)
        good = np.all(np.isfinite(dx), axis=1) & (lengths > 0.0)

        out = np.full(a1 - a0, np.nan, float)
        out[(~good) & (lengths == 0.0)] = 0.0
        if np.any(good):
            Q = p0[good, None, :] + ts[None, :, None] * dx[good, None, :]
            Qflat = Q.reshape(-1, 3)
            gx, gy, gz = _grad_Umb_fd_3d(Qflat, p2, p3, fd_eps=fd_eps)
            G = np.column_stack([gx, gy, gz]).reshape(np.count_nonzero(good), qn, 3)
            integ = np.einsum("bqd,bd->bq", G, dx[good])
            vals = np.sum(integ * wts[None, :], axis=1)
            out[good] = vals
        dU_edges[a0:a1] = out

    return dU_edges

def path_integrated_U_on_known_nodes_3d(
    known_nodes,
    anchor_g,
    P_full,
    n12,
    n13,
    n23,
    feas_mask_flat,
    p2,
    p3,
    graph_mode=None,
    knn_k=None,
    knn_max_dist=None,
    extra_edges=None,
    quad_order=16,
    fd_eps=1e-4,
):
    """Assign nodal energies by on-the-fly path integration of exact oracle gradients.

    This mirrors path_integrated_U_on_known_nodes_2d(...).  It does not use
    LSQR and does not read oracle energies for training labels.  It uses the
    current paid graph, computes local line integrals along paid-node edges,
    and propagates energies from the anchor along shortest graph paths.
    """
    known_nodes = np.array(sorted(set(map(int, known_nodes))), dtype=int)
    known_nodes = known_nodes[feas_mask_flat[known_nodes]]
    nk = known_nodes.size
    if nk == 0:
        return np.array([], float), known_nodes

    loc = {int(g): i for i, g in enumerate(known_nodes)}
    if int(anchor_g) not in loc:
        return np.full(nk, np.nan, float), known_nodes

    mode = (LSQR_GRAPH_MODE if graph_mode is None else str(graph_mode)).lower()

    if mode == "knn":
        k_use = int(knn_k if knn_k is not None else LSQR_KNN_K)
        r_use = knn_max_dist if knn_max_dist is not None else LSQR_KNN_MAX_DIST
        edges = build_edges_knn_from_nodes(known_nodes, P_full, k=k_use, max_dist=r_use)
    elif mode == "axis1":
        edges = build_edges_axis1_from_nodes(known_nodes, n12, n13, n23, feas_mask_flat)
    else:
        edges = build_edges_from_nodes(known_nodes, n12, n13, n23, feas_mask_flat)

    if extra_edges:
        es = set((int(a), int(b)) if int(a) < int(b) else (int(b), int(a)) for (a, b) in edges)
        for (a, b) in extra_edges:
            a = int(a); b = int(b)
            if a == b:
                continue
            if (a not in loc) or (b not in loc):
                continue
            if (not feas_mask_flat[a]) or (not feas_mask_flat[b]):
                continue
            i, j = (a, b) if a < b else (b, a)
            es.add((i, j))
        edges = list(es)

    adj = [[] for _ in range(nk)]
    n_edges = 0
    n_bad = 0

    for ii, jj in edges:
        ii = int(ii); jj = int(jj)
        if (ii not in loc) or (jj not in loc):
            continue
        li = loc[ii]
        lj = loc[jj]
        length = float(np.linalg.norm(_ala_displacement(P_full[jj],P_full[ii]) if ALANINE_MODE else P_full[jj]-P_full[ii]))
        dU_ij = _edge_work_quadrature_3d(
            P_full[ii], P_full[jj], p2, p3,
            quad_order=quad_order,
            fd_eps=fd_eps,
        )
        if not np.isfinite(dU_ij):
            n_bad += 1
            continue
        adj[li].append((lj, length, dU_ij))
        adj[lj].append((li, length, -dU_ij))
        n_edges += 1

    if LSQR_PRINT_EQUATION_STATS:
        print(
            f"[PATHINT build 3D] mode={mode} nk={nk} edges={n_edges} "
            f"bad_edges={n_bad} quad={quad_order} fd_eps={fd_eps}"
        )

    a_loc = loc[int(anchor_g)]
    dist = np.full(nk, np.inf, float)
    U = np.full(nk, np.nan, float)
    dist[a_loc] = 0.0
    U[a_loc] = 0.0
    pq = [(0.0, a_loc)]

    while pq:
        dcur, u = heapq.heappop(pq)
        if dcur != dist[u]:
            continue
        for v, length, dU_uv in adj[u]:
            nd = dcur + length
            if nd < dist[v]:
                dist[v] = nd
                U[v] = U[u] + dU_uv
                heapq.heappush(pq, (nd, v))

    U -= U[a_loc]
    if ALANINE_MODE:_ALA.record_labels(known_nodes,anchor_g,P_full,U,adj)
    return U, known_nodes

# =============================================================================
# Step-0 geometry builders (unchanged behavior)
# =============================================================================
def cube_edge_nodes(feasible_g, n12, n13, n23):
    if ALANINE_MODE:
        phi=np.linspace(-np.pi,np.pi,n13,endpoint=False)
        psi=np.linspace(-np.pi,np.pi,n23,endpoint=False)
        i=int(np.argmin(np.abs(wrapped(phi-_ALA.anchor_rad[0]))))
        j=int(np.argmin(np.abs(wrapped(psi-_ALA.anchor_rad[1]))))
        return np.array([ijk_to_g(0,i,j,n12,n13,n23)],int)
    out = []
    for g in feasible_g:
        i12, i13, i23 = g_to_ijk(int(g), n12, n13, n23)
        b12 = (i12 == 0) or (i12 == n12 - 1)
        b13 = (i13 == 0) or (i13 == n13 - 1)
        b23 = (i23 == 0) or (i23 == n23 - 1)
        if (b12 + b13 + b23) >= 2:
            out.append(int(g))
    return np.array(sorted(set(out)), dtype=int)

def _line_nodes_edges_between(i12, i13, i23, j12, j13, j23, n12, n13, n23, feas_mask_flat):
    i12 = int(i12); i13 = int(i13); i23 = int(i23)
    j12 = int(j12); j13 = int(j13); j23 = int(j23)
    s12 = 0 if j12 == i12 else (1 if j12 > i12 else -1)
    s13 = 0 if j13 == i13 else (1 if j13 > i13 else -1)
    s23 = 0 if j23 == i23 else (1 if j23 > i23 else -1)
    steps = max(abs(j12 - i12), abs(j13 - i13), abs(j23 - i23))
    nodes = []
    for t in range(steps + 1):
        a12 = i12 + s12 * t
        a13 = i13 + s13 * t
        a23 = i23 + s23 * t
        if 0 <= a12 < n12 and 0 <= a13 < n13 and 0 <= a23 < n23:
            g = ijk_to_g(a12, a13, a23, n12, n13, n23)
            if feas_mask_flat[int(g)]:
                nodes.append(int(g))
    seen = set()
    out_nodes = []
    for g in nodes:
        if g not in seen:
            seen.add(g); out_nodes.append(g)
    out_nodes = np.array(out_nodes, dtype=int)
    out_edges = []
    for a, b in zip(out_nodes[:-1], out_nodes[1:]):
        i, j = (a, b) if a < b else (b, a)
        if i != j:
            out_edges.append((int(i), int(j)))
    return out_nodes, out_edges

def _all_cube_edges_ijk(n12, n13, n23):
    lo12, hi12 = 0, n12-1
    lo13, hi13 = 0, n13-1
    lo23, hi23 = 0, n23-1
    edges = []
    for a13 in (lo13, hi13):
        for a23 in (lo23, hi23):
            edges.append(((lo12, a13, a23), (hi12, a13, a23)))
    for a12 in (lo12, hi12):
        for a23 in (lo23, hi23):
            edges.append(((a12, lo13, a23), (a12, hi13, a23)))
    for a12 in (lo12, hi12):
        for a13 in (lo13, hi13):
            edges.append(((a12, a13, lo23), (a12, a13, hi23)))
    seen = set(); uniq = []
    for p, q in edges:
        key = tuple(p) + tuple(q) if tuple(p) < tuple(q) else tuple(q) + tuple(p)
        if key not in seen:
            seen.add(key); uniq.append((p, q))
    return uniq

def _select_connected_cube_edges_ijk(n12, n13, n23, anchor_ijk, edge_count):
    edge_count = int(max(0, min(12, edge_count)))
    if edge_count == 0:
        return []

    lo12, hi12 = 0, n12 - 1
    lo13, hi13 = 0, n13 - 1
    lo23, hi23 = 0, n23 - 1

    a12, a13, a23 = map(int, anchor_ijk)

    def _snap(i, lo, hi):
        if i <= lo:
            return lo
        if i >= hi:
            return hi
        return hi

    start = (_snap(a12, lo12, hi12), _snap(a13, lo13, hi13), _snap(a23, lo23, hi23))

    edges = _all_cube_edges_ijk(n12, n13, n23)
    if edge_count >= len(edges):
        return edges[:]

    adj = {}
    for (p, q) in edges:
        p = tuple(map(int, p)); q = tuple(map(int, q))
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)

    from collections import deque
    visited = {start}
    depth = {start: 0}
    q = deque([start])

    selected = []
    selected_set = set()

    def _edge_key(p, q):
        p = tuple(p); q = tuple(q)
        return (p, q) if p < q else (q, p)

    while q and len(selected) < min(edge_count, 7):
        u = q.popleft()
        nbrs = sorted(adj.get(u, []))
        for v in nbrs:
            ekey = _edge_key(u, v)
            if ekey in selected_set:
                continue
            selected.append((u, v))
            selected_set.add(ekey)
            if v not in visited:
                visited.add(v)
                depth[v] = depth[u] + 1
                q.append(v)
            if len(selected) >= min(edge_count, 7):
                break

    if len(selected) >= edge_count:
        return selected[:edge_count]

    remaining = []
    for (p, q2) in edges:
        ekey = _edge_key(p, q2)
        if ekey in selected_set:
            continue
        dp = depth.get(tuple(p), 10**9)
        dq = depth.get(tuple(q2), 10**9)
        remaining.append((min(dp, dq), ekey, (tuple(p), tuple(q2))))
    remaining.sort(key=lambda t: (t[0], t[1]))

    for _, _, (p, q2) in remaining:
        selected.append((p, q2))
        if len(selected) >= edge_count:
            break

    return selected[:edge_count]

def _all_face_diagonals_ijk(n12, n13, n23):
    lo12, hi12 = 0, n12-1
    lo13, hi13 = 0, n13-1
    lo23, hi23 = 0, n23-1
    diags = []
    for a12 in (lo12, hi12):
        diags.append(((a12, lo13, lo23), (a12, hi13, hi23)))
        diags.append(((a12, lo13, hi23), (a12, hi13, lo23)))
    for a13 in (lo13, hi13):
        diags.append(((lo12, a13, lo23), (hi12, a13, hi23)))
        diags.append(((lo12, a13, hi23), (hi12, a13, lo23)))
    for a23 in (lo23, hi23):
        diags.append(((lo12, lo13, a23), (hi12, hi13, a23)))
        diags.append(((lo12, hi13, a23), (hi12, lo13, a23)))
    seen = set(); uniq = []
    for p, q in diags:
        key = tuple(p) + tuple(q) if tuple(p) < tuple(q) else tuple(q) + tuple(p)
        if key not in seen:
            seen.add(key); uniq.append((p, q))
    return uniq

def _all_space_diagonals_ijk(n12, n13, n23):
    lo12, hi12 = 0, n12-1
    lo13, hi13 = 0, n13-1
    lo23, hi23 = 0, n23-1
    return [
        ((lo12, lo13, lo23), (hi12, hi13, hi23)),
        ((lo12, lo13, hi23), (hi12, hi13, lo23)),
        ((lo12, hi13, lo23), (hi12, lo13, hi23)),
        ((lo12, hi13, hi23), (hi12, lo13, lo23)),
    ]

def _rank_lines_by_anchor_endpoint(lines, anchor_ijk):
    a = tuple(map(int, anchor_ijk))
    def key_fn(pair):
        p, q = pair
        p = tuple(p); q = tuple(q)
        hit = (p == a) or (q == a)
        lo, hi = (p, q) if p < q else (q, p)
        return (0 if hit else 1, lo, hi)
    return sorted(lines, key=key_fn)

def _build_step0_geometry_from_counts(anchor_g, n12, n13, n23, feas_mask_flat,
                                      edge_count, face_diag_count, space_diag_count):
    if ALANINE_MODE:
        return _ala_seed_geometry(anchor_g,n12,n13,n23)
    edge_count = int(np.clip(edge_count, 0, 12))
    face_diag_count = int(np.clip(face_diag_count, 0, 12))
    space_diag_count = int(np.clip(space_diag_count, 0, 4))

    a12, a13, a23 = g_to_ijk(int(anchor_g), n12, n13, n23)
    anchor_ijk = (a12, a13, a23)

    nodes_all = set([int(anchor_g)])
    extra_edges = set()

    all_edges = _select_connected_cube_edges_ijk(n12, n13, n23, anchor_ijk, edge_count)
    for (p, q) in all_edges:
        n, e = _line_nodes_edges_between(*p, *q, n12, n13, n23, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for (i, j) in e:
            extra_edges.add((i, j) if i < j else (j, i))

    all_face = _rank_lines_by_anchor_endpoint(_all_face_diagonals_ijk(n12, n13, n23), anchor_ijk)
    for (p, q) in all_face[:face_diag_count]:
        n, e = _line_nodes_edges_between(*p, *q, n12, n13, n23, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for (i, j) in e:
            extra_edges.add((i, j) if i < j else (j, i))

    all_space = _all_space_diagonals_ijk(n12, n13, n23)
    a = tuple(anchor_ijk)
    anchor_space = [pq for pq in all_space if (tuple(pq[0]) == a or tuple(pq[1]) == a)]
    others_space = [pq for pq in all_space if pq not in anchor_space]

    chosen_space = []
    if (space_diag_count >= 1) and (len(anchor_space) == 1):
        chosen_space.append(anchor_space[0])
        chosen_space.extend(sorted(others_space)[:max(0, space_diag_count - 1)])
    else:
        chosen_space = sorted(all_space)[:space_diag_count]

    for (p, q) in chosen_space:
        n, e = _line_nodes_edges_between(*p, *q, n12, n13, n23, feas_mask_flat)
        nodes_all.update(map(int, n.tolist()))
        for (i, j) in e:
            extra_edges.add((i, j) if i < j else (j, i))

    return np.array(sorted(nodes_all), dtype=int), list(extra_edges)

# =============================================================================
# STEP-0 CONNECTIVITY FIX
# =============================================================================
def _components_in_step0(nodes, n12, n13, n23, feas_flat):
    nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
    node_set = set(nodes.tolist())
    seen = set()
    comps = []
    for g in nodes:
        g = int(g)
        if g in seen:
            continue
        if not feas_flat[g]:
            continue
        stack = [g]
        seen.add(g)
        comp = []
        while stack:
            u = int(stack.pop())
            comp.append(u)
            for v in neighbors_26(u, n12, n13, n23):
                v = int(v)
                if v in seen:
                    continue
                if (v in node_set) and feas_flat[v]:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps

def enforce_step0_connected(step0_nodes, anchor_g, P_full, n12, n13, n23, feas_flat):
    anchor_g = int(anchor_g)
    step0_nodes = np.array(sorted(set(map(int, step0_nodes))), dtype=int)
    if step0_nodes.size == 0:
        return step0_nodes

    nodes_set = set(step0_nodes.tolist())
    nodes_set.add(anchor_g)

    comps = _components_in_step0(nodes_set, n12, n13, n23, feas_flat)
    comp_anchor = None
    for comp in comps:
        if anchor_g in comp:
            comp_anchor = comp
            break
    if comp_anchor is None:
        comp_anchor = [anchor_g]
        comps = [comp_anchor] + comps

    if len(comps) <= 1:
        print(f"[STEP0 connect] components=1 (already connected)")
        return np.array(sorted(nodes_set), dtype=int)

    anchor_pt = P_full[anchor_g]
    train_set = set(nodes_set)

    sizes = [len(c) for c in comps]
    print(f"[STEP0 connect] components={len(comps)} sizes={sorted(sizes, reverse=True)[:8]}{'...' if len(comps)>8 else ''}")

    for comp in comps:
        if anchor_g in comp:
            continue
        comp = np.array(comp, dtype=int)
        pts = P_full[comp]
        j = int(np.argmin(np.linalg.norm(pts - anchor_pt[None, :], axis=1)))
        g_rep = int(comp[j])
        added = shortest_path_add_3d(train_set, g_rep, anchor_g, P_full, n12, n13, n23, feas_flat)
        if len(added) == 0:
            _ = manhattan_walk_add_3d(train_set, g_rep, anchor_g, n12, n13, n23, feas_flat)

    out = np.array(sorted(train_set), dtype=int)
    comps2 = _components_in_step0(out, n12, n13, n23, feas_flat)
    print(f"[STEP0 connect] after-bridge components={len(comps2)}  nodes={out.size}")
    return out

# =============================================================================
# Overlay MST helpers
# =============================================================================
def mst_edges_over_allowed_edges(P, nodes, allowed_edges):
    nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
    if nodes.size < 2:
        return []
    pos = {int(g): i for i, g in enumerate(nodes)}
    rows, cols, data = [], [], []
    for (gi, gj) in allowed_edges:
        gi = int(gi); gj = int(gj)
        ai = pos.get(gi, None); aj = pos.get(gj, None)
        if ai is None or aj is None:
            continue
        w = float(np.linalg.norm(_ala_displacement(P[gi],P[gj]) if ALANINE_MODE else P[gi]-P[gj]))
        rows += [ai, aj]; cols += [aj, ai]; data += [w, w]
    if len(data) == 0:
        return []
    m = nodes.size
    G = coo_matrix((np.asarray(data, float),
                    (np.asarray(rows, int), np.asarray(cols, int))),
                   shape=(m, m)).tocsr()
    T = minimum_spanning_tree(G).tocoo()
    out = []
    for a, b in zip(T.row, T.col):
        out.append((int(nodes[int(a)]), int(nodes[int(b)])))
    return out

def overlay_train_val_mst_factory(Pall, train_all_idx, train_q_idx, val_idx, anchor_idx,
                                  knn_idx=None, knn_score_map=None,
                                  mst_allowed_edges_all=None, mst_allowed_edges_q=None,
                                  scatter_swap_xy=False,
                                  step0_overlay_edges=None):

    train_all_idx = np.array(sorted(set(map(int, train_all_idx))), dtype=int)
    train_q_idx   = np.array(sorted(set(map(int, train_q_idx))), dtype=int)
    val_idx       = np.array(sorted(set(map(int, val_idx))), dtype=int)
    knn_idx = np.array([], dtype=int) if knn_idx is None else np.array(sorted(set(map(int, knn_idx))), dtype=int)

    if step0_overlay_edges:
        step0_overlay_edges = list({
            (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            for (a, b) in step0_overlay_edges
            if int(a) != int(b)
        })
    else:
        step0_overlay_edges = []

    def _plot_mst_3d(ax, idxs, allowed_edges, color, lw, alpha, zorder=9999):
        idxs = np.array(sorted(set(map(int, idxs))), dtype=int)
        if idxs.size < 2:
            return
        if allowed_edges is not None and len(allowed_edges) > 0:
            mstE = mst_edges_over_allowed_edges(Pall, idxs, allowed_edges)
            for (gi, gj) in mstE:
                pts = _apply_axis_mapping(Pall[[gi, gj]], swap_xy=scatter_swap_xy)
                ax.plot([pts[0, 0], pts[1, 0]],
                        [pts[0, 1], pts[1, 1]],
                        [pts[0, 2], pts[1, 2]],
                        color=color, lw=lw, alpha=alpha,
                        solid_capstyle="round", clip_on=False, zorder=zorder)
        else:
            pts = _apply_axis_mapping(Pall[idxs], swap_xy=scatter_swap_xy)
            D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            G = minimum_spanning_tree(coo_matrix(D)).tocoo()
            for i, j in zip(G.row, G.col):
                ax.plot([pts[i, 0], pts[j, 0]],
                        [pts[i, 1], pts[j, 1]],
                        [pts[i, 2], pts[j, 2]],
                        color=color, lw=lw, alpha=alpha,
                        solid_capstyle="round", clip_on=False, zorder=zorder)

    def overlay(ax):
        Pm = _apply_axis_mapping(Pall, swap_xy=scatter_swap_xy)

        def draw_train():
            tr = Pm[train_all_idx]
            ax.scatter(tr[:, 0], tr[:, 1], tr[:, 2],
                       s=MS_TRAIN, marker=TRAIN_MARKER,
                       facecolors=TRAIN_FACE, edgecolors=TRAIN_EDGE,
                       linewidths=0.8, alpha=1.0,
                       depthshade=False, zorder=Z_TRAIN)

        def draw_val():
            if val_idx.size > 0:
                vv = val_idx[:min(15000, val_idx.size)]
                v = Pm[vv]
                ax.scatter(v[:, 0], v[:, 1], v[:, 2],
                           s=MS_VAL, marker=VAL_MARKER,
                           facecolors=VAL_FACE, edgecolors=VAL_EDGE,
                           linewidths=0.9, alpha=0.12,
                           depthshade=False, zorder=Z_VAL)

        def draw_anchor():
            a = Pm[int(anchor_idx)]
            ax.scatter([a[0]], [a[1]], [a[2]],
                       s=MS_ANCHOR, marker="*", c="gold",
                       edgecolors="k", linewidths=1.0, alpha=1.0,
                       depthshade=False, zorder=Z_ANCHOR)

        knn_mappable = None
        def draw_knn():
            nonlocal knn_mappable
            knn_mappable = None
            if SHOW_CAND_OVERLAY and knn_idx.size > 0 and isinstance(knn_score_map, dict):
                cs = np.asarray([knn_score_map.get(int(g), np.nan) for g in knn_idx], float)
                msk = np.isfinite(cs)
                if np.any(msk):
                    pts = Pm[knn_idx[msk]]
                    cs2 = cs[msk]
                    v0 = float(np.percentile(cs2, 5))
                    v1 = float(np.percentile(cs2, 95))
                    if (not np.isfinite(v0)) or (not np.isfinite(v1)) or (v1 <= v0):
                        v0 = float(np.nanmin(cs2)); v1 = float(np.nanmax(cs2))
                    cnorm = mpl.colors.Normalize(vmin=v0, vmax=v1)
                    knn_mappable = ax.scatter(
                        pts[:, 0], pts[:, 1], pts[:, 2],
                        s=CAND_MARKER_SIZE, c=cs2,
                        cmap=CAND_CMAP, norm=cnorm,
                        edgecolors="k", linewidths=CAND_EDGE_LW,
                        alpha=CAND_ALPHA, depthshade=False, zorder=Z_KNN
                    )

        def draw_mst_all():
            _plot_mst_3d(ax, train_all_idx, mst_allowed_edges_all, MST_COLOR, LINE_MST_WIDTH, 0.95, zorder=12000)

        def draw_mst_q():
            _plot_mst_3d(ax, train_q_idx, mst_allowed_edges_q, MST_Q_COLOR, LINE_MST_WIDTH * 1.15, 0.98, zorder=13000)

        def draw_step0_geometry_edges():
            if not step0_overlay_edges:
                return
            for (gi, gj) in step0_overlay_edges:
                pts = _apply_axis_mapping(Pall[[gi, gj]], swap_xy=scatter_swap_xy)
                ax.plot([pts[0, 0], pts[1, 0]],
                        [pts[0, 1], pts[1, 1]],
                        [pts[0, 2], pts[1, 2]],
                        color="k", lw=LINE_MST_WIDTH * 1.35, alpha=1.0,
                        solid_capstyle="round", clip_on=False, zorder=20000)

        if OVERLAY_ORDER_MODE == "mst_first":
            order = ["mst_all", "mst_q", "train", "val", "anchor", "knn", "step0geom"]
        else:
            order = ["train", "val", "anchor", "knn", "mst_all", "mst_q", "step0geom"]

        dispatch = {
            "mst_all": draw_mst_all, "mst_q": draw_mst_q,
            "train": draw_train, "val": draw_val,
            "anchor": draw_anchor, "knn": draw_knn,
            "step0geom": draw_step0_geometry_edges,
        }
        for key in order:
            fn = dispatch.get(key, None)
            if fn is not None:
                fn()

        if knn_mappable is not None:
            return {"knn_mappable": knn_mappable, "knn_label": CAND_CBAR_LABEL}
        return None

    overlay.figure_data = {
        "Pall": Pall,
        "train_all_idx": train_all_idx,
        "train_q_idx": train_q_idx,
        "val_idx": val_idx,
        "anchor_idx": int(anchor_idx),
        "knn_idx": knn_idx,
        "knn_score_map": knn_score_map,
        "mst_allowed_edges_all": mst_allowed_edges_all,
        "mst_allowed_edges_q": mst_allowed_edges_q,
        "step0_overlay_edges": step0_overlay_edges,
    }
    return overlay

# =============================================================================
# Metrics plots
# =============================================================================
def plot_rmse_history_each_step(rmse_tr_hist, rmse_val_hist, step_name, target_label, train_set_label="paid"):
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    its = np.arange(0, len(rmse_tr_hist))
    ax.plot(its, rmse_tr_hist, "-", lw=5, label=f"Train")
    ax.plot(its, rmse_val_hist, "-", lw=5, label=("External FES diagnostic" if ALANINE_MODE else "Val"))
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("RMSE", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(frameon=False, fontsize=LEGEND_FZ, loc="upper right",
          markerscale=2.0, handlelength=0.5, handletextpad=0.6,
          labelspacing=0.5, borderpad=0.2, borderaxespad=0.2)
    fig_name = f"{target_label}_AL_step{step_name}_rmse_trace"
    save_numeric_figure_data(
        fig_name,
        np.column_stack([its, np.asarray(rmse_tr_hist, float), np.asarray(rmse_val_hist, float)]),
        "al_step,rmse_train,rmse_val",
    )
    savefig(fig, fig_name)

def plot_mst_history_each_step(mst_hist, step_name):
    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    steps = np.arange(len(mst_hist))
    ax.plot(steps, mst_hist, "-o", lw=5, ms=16, mfc="none")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("MST length", fontsize=AXIS_LABEL_FZ, labelpad=18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    fig_name = f"W3_AL_step{step_name}_mst_trace"
    save_numeric_figure_data(fig_name, np.column_stack([steps, np.asarray(mst_hist, float)]), "al_step,mst_length")
    savefig(fig, fig_name)

def plot_parity_each_step(model, X_tr, y_tr, X_val, y_val, ymu, ysd, step_name, target_label, train_set_label="paid"):
    mu_tr_z, _ = predict_mu_std_z(model, X_tr)
    mu_val_z, _ = predict_mu_std_z(model, X_val)
    yhat_tr = z2y(mu_tr_z, ymu, ysd)
    yhat_val = z2y(mu_val_z, ymu, ysd)

    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    ax.scatter(y_tr, yhat_tr, s=160, alpha=1, label=f"Train")
    ax.scatter(y_val, yhat_val, s=160, alpha=0.35, edgecolors="none", label="Val")

    allv = np.concatenate([y_tr[np.isfinite(y_tr)], y_val[np.isfinite(y_val)],
                           yhat_tr[np.isfinite(yhat_tr)], yhat_val[np.isfinite(yhat_val)]])
    if allv.size > 0:
        vmin = float(np.nanpercentile(allv, 1))
        vmax = float(np.nanpercentile(allv, 99))
    else:
        vmin, vmax = -1.0, 1.0

    ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=2.0, alpha=0.7)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r"$W_3^{\mathrm{True}}\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(r"$W_3^{\mathrm{Pred}}\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=AXIS_TICK_PAD_2D,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.legend(frameon=False, fontsize=LEGEND_FZ, loc="upper left",
          markerscale=1.5, handlelength=0.2, handletextpad=0.6,
          labelspacing=0.5, borderpad=0.2, borderaxespad=0.2)
    fig_name = f"{target_label}_AL_step{step_name}_parity"
    parity_data = np.row_stack(
        [
            np.column_stack([np.zeros(len(y_tr)), y_tr, yhat_tr]),
            np.column_stack([np.ones(len(y_val)), y_val, yhat_val]),
        ]
    )
    save_numeric_figure_data(fig_name, parity_data, "set_code_0_train_1_val,true_value,predicted_value")
    savefig(fig, fig_name)

def plot_lsqr_paid_vs_fullgrid_parity(U_paid, paid_nodes, U_full_map, step_name):
    paid_nodes = np.array(paid_nodes, dtype=int)
    u_paid = np.asarray(U_paid, float).reshape(-1)
    u_full = np.asarray(U_full_map[paid_nodes], float).reshape(-1)
    m = np.isfinite(u_paid) & np.isfinite(u_full)
    if not np.any(m):
        return
    u_paid = u_paid[m]; u_full = u_full[m]
    r = rmse(u_full, u_paid)
    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    ax.scatter(u_full, u_paid, s=22, alpha=0.55, edgecolors="none")
    vmin = float(np.nanpercentile(np.r_[u_full, u_paid], 1))
    vmax = float(np.nanpercentile(np.r_[u_full, u_paid], 99))
    ax.plot([vmin, vmax], [vmin, vmax], "k-", lw=2.0, alpha=0.7)
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel(r"Full-grid LSQR $W_3\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel(r"On-the-fly LSQR $W_3\,(k_{\mathrm{B}}T)$", fontsize=AXIS_LABEL_FZ)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FZ, pad=AXIS_TICK_PAD_2D,
                   length=AXIS_TICK_LENGTH, width=AXIS_TICK_WIDTH)
    ax.text(0.02, 0.98, f"RMSE = {r:.4f}\nN = {u_full.size}",
            transform=ax.transAxes, va="top", ha="left", fontsize=LEGEND_FZ)
    fig_name = f"LSQR_AL_step{step_name}_paid_vs_fullgrid_parity"
    save_numeric_figure_data(fig_name, np.column_stack([u_full, u_paid]), "full_grid_lsqr,on_the_fly_lsqr")
    savefig(fig, fig_name)

# =============================================================================
# Force computation on masked feasible region (for LSQR teacher forces)
# =============================================================================
def fill_infeasible_by_nearest(Ugrid_nan, feas_mask, d12, d13, d23):
    Ugrid_nan = np.asarray(Ugrid_nan, float)
    feas_mask = np.asarray(feas_mask, bool)

    if not np.any(feas_mask):
        return np.zeros_like(Ugrid_nan)

    D12c, D13c, D23c = np.meshgrid(d12, d13, d23, indexing="ij")
    coords = np.c_[D12c[feas_mask].ravel(), D13c[feas_mask].ravel(), D23c[feas_mask].ravel()]
    vals   = Ugrid_nan[feas_mask].ravel()

    tree = cKDTree(coords)

    nan_mask = ~feas_mask
    if not np.any(nan_mask):
        return Ugrid_nan.copy()

    qcoords = np.c_[D12c[nan_mask].ravel(), D13c[nan_mask].ravel(), D23c[nan_mask].ravel()]
    _, nn = tree.query(qcoords, k=1)
    filled = Ugrid_nan.copy()
    filled[nan_mask] = vals[nn]
    filled = np.nan_to_num(filled, nan=0.0)
    return filled

def compute_teacher_forces(Ugrid_nan, feas, d12, d13, d23):
    if ALANINE_MODE:
        return tuple(np.zeros_like(Ugrid_nan) for _ in range(3))
    U_filled = fill_infeasible_by_nearest(Ugrid_nan, feas, d12, d13, d23)

    dd12 = float(d12[1] - d12[0]) if len(d12) > 1 else 1.0
    dd13 = float(d13[1] - d13[0]) if len(d13) > 1 else 1.0
    dd23 = float(d23[1] - d23[0]) if len(d23) > 1 else 1.0

    dU_dd12, dU_dd13, dU_dd23 = np.gradient(U_filled, dd12, dd13, dd23, edge_order=2)

    dU_dd12 = np.where(feas, dU_dd12, np.nan)
    dU_dd13 = np.where(feas, dU_dd13, np.nan)
    dU_dd23 = np.where(feas, dU_dd23, np.nan)
    return dU_dd12, dU_dd13, dU_dd23

# =============================================================================
# Per-step full plotting
# =============================================================================
def plot_all_figs_for_step(step_name, model, plot_g, P_full, P_feas, feas_pos, Xs_feas,
                           xmu, xsd, ymu, ysd, normU,
                           U_oracle_flat, U_oracle_diag, U_oracle_face,
                           train_all_set, train_q_set, val_g, anchor_g,
                           d12s, d13s, D12s, D13s, D23s, mask2,
                           d13f, d23f, D13f, D23f, D12f, maskf,
                           p2, W2_anchor,
                           score_map_for_this_step=None, cand_overlay_idx_for_this_step=None,
                           X_tr=None, y_tr=None, X_val=None, y_val=None,
                           rmse_tr_hist=None, rmse_val_hist=None, mst_hist=None,
                           n12=None, n13=None, n23=None, feas_flat=None,
                           step0_extra_edges=None,
                           step0_overlay_edges=None,
                           target_mode="umb",
                           W2anch_feas=None,
                           target_label="W3",
                           symmetry_mode="sorted",
                           train_set_label="paid",
                           U_full_map=None, paid_nodes_step=None, U_paid_step=None):

    if ALANINE_MODE:
        return _ala_plot_step(step_name,model,plot_g,P_full,P_feas,feas_pos,Xs_feas,
            xmu,xsd,ymu,ysd,normU,U_oracle_flat,U_oracle_diag,U_oracle_face,
            train_all_set,train_q_set,val_g,anchor_g,
            score_map_for_this_step=score_map_for_this_step,
            cand_overlay_idx_for_this_step=cand_overlay_idx_for_this_step)
    if cand_overlay_idx_for_this_step is None:
        cand_overlay_idx_for_this_step = np.array([], dtype=int)

    mst_edges_all = None
    mst_edges_q = None
    if (n12 is not None) and (n13 is not None) and (n23 is not None) and (feas_flat is not None):
        mode_use = STEP0_LSQR_GRAPH_MODE if str(step_name) == "00" else LSQR_GRAPH_MODE
        mode = str(mode_use).lower()

        if mode == "axis1":
            mst_edges_all = build_edges_axis1_from_nodes(train_all_set, n12, n13, n23, feas_flat)
            mst_edges_q   = build_edges_axis1_from_nodes(train_q_set,   n12, n13, n23, feas_flat)
            if step0_extra_edges:
                mst_edges_all = list(set(tuple(sorted(e)) for e in (mst_edges_all + step0_extra_edges)))
                mst_edges_q   = list(set(tuple(sorted(e)) for e in (mst_edges_q   + step0_extra_edges)))
        elif mode == "knn":
            mst_edges_all = build_edges_knn_from_nodes(train_all_set, P_full, k=LSQR_KNN_K, max_dist=LSQR_KNN_MAX_DIST)
            mst_edges_q   = build_edges_knn_from_nodes(train_q_set,   P_full, k=LSQR_KNN_K, max_dist=LSQR_KNN_MAX_DIST)
        else:
            mst_edges_all = build_edges_from_nodes(train_all_set, n12, n13, n23, feas_flat)
            mst_edges_q   = build_edges_from_nodes(train_q_set,   n12, n13, n23, feas_flat)

    overlay = overlay_train_val_mst_factory(
        P_full,
        train_all_idx=np.array(sorted(train_all_set), int),
        train_q_idx=np.array(sorted(train_q_set), int),
        val_idx=val_g,
        anchor_idx=anchor_g,
        knn_idx=cand_overlay_idx_for_this_step,
        knn_score_map=score_map_for_this_step,
        mst_allowed_edges_all=mst_edges_all,
        mst_allowed_edges_q=mst_edges_q,
        scatter_swap_xy=False,
        step0_overlay_edges=step0_overlay_edges
    )

    if (U_full_map is not None) and (paid_nodes_step is not None) and (U_paid_step is not None):
        plot_lsqr_paid_vs_fullgrid_parity(U_paid_step, paid_nodes_step, U_full_map, step_name)

    mu_z, std_z = predict_mu_std_z(model, Xs_feas)
    mu_target = z2y(mu_z, ymu, ysd)
    std_target = std_z * ysd

    if str(target_mode).lower() == "residual":
        if W2anch_feas is None:
            raise RuntimeError("W2anch_feas must be provided when target_mode='residual'.")
        mu = mu_target + W2anch_feas
    else:
        mu = mu_target

    plot_pos = feas_pos[plot_g]

    plot_scatter3d_color(
        P_feas[plot_pos], mu[plot_pos], LAB_C,
        f"W3_AL_step{step_name}_mu_scatter",
        title=rf"Student $\mu$ (AL step {step_name}), {GRAFT}",
        cmap="viridis", norm=normU, overlay=overlay,
        scatter_swap_xy=False
    )

    mu2_t, sig2 = gp_predict_on_slice(model, xmu, xsd, ymu, ysd, D12s, D13s, symmetry_mode)
    mu_face_t, _ = gp_predict_on_face(model, xmu, xsd, ymu, ysd, D13f, D23f, D12f, symmetry_mode)

    if str(target_mode).lower() == "residual":
        W2sum_diag = W2sum_pairs(D12s, D13s, D23s, p2)
        mu2 = mu2_t + (W2sum_diag - W2_anchor)
        W2sum_face = W2sum_pairs(D12f, D13f, D23f, p2)
        mu_face = mu_face_t + (W2sum_face - W2_anchor)
    else:
        mu2 = mu2_t
        mu_face = mu_face_t

    err_scatter = _abs_err(mu[plot_pos], U_oracle_flat[plot_g])
    err_diag = _abs_err(mu2, U_oracle_diag)
    err_face = _abs_err(mu_face, U_oracle_face)

    err_vals = []

    e0 = err_scatter[np.isfinite(err_scatter)]
    if e0.size > 0:
        err_vals.append(e0)

    e2 = err_diag[mask2 & np.isfinite(err_diag)]
    if e2.size > 0:
        err_vals.append(e2)

    ef = err_face[maskf & np.isfinite(err_face)]
    if ef.size > 0:
        err_vals.append(ef)

    if len(err_vals) > 0:
        err_all = np.concatenate(err_vals)
        err_vmax = float(np.nanpercentile(err_all, 99.0))
        if (not np.isfinite(err_vmax)) or (err_vmax <= 0.0):
            err_vmax = float(np.nanmax(err_all)) if err_all.size > 0 else 1.0
        if (not np.isfinite(err_vmax)) or (err_vmax <= 0.0):
            err_vmax = 1.0
    else:
        err_vmax = 1.0

    normErr = mpl.colors.Normalize(vmin=0.0, vmax=err_vmax)

    plot_scatter3d_color(
        P_feas[plot_pos], err_scatter, LAB_ABSERR,
        f"W3_AL_step{step_name}_abserr_scatter",
        title=rf"Absolute error scatter, step {step_name}, {GRAFT}",
        cmap=ABSERR_CMAP, norm=None, overlay=overlay,
        scatter_swap_xy=False
    )

    normS = mpl.colors.Normalize(vmin=float(np.nanmin(std_target)), vmax=float(np.nanmax(std_target)))
    plot_scatter3d_color(
        P_feas[plot_pos], std_target[plot_pos], r"$\sigma^{\mathrm{Pred}}$",
        f"W3_AL_step{step_name}_sigma_scatter",
        title=rf"Student $\sigma$ (AL step {step_name}), {GRAFT}",
        cmap="plasma", norm=normS, overlay=overlay,
        scatter_swap_xy=False
    )

    plot_slice_2d_heat(D12s, D13s, mu2, mask2,
        f"W3_AL_step{step_name}_mu_slice2D",
        title=rf"Student $\mu$ diag slice ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )
    plot_slice_3d_surface(D12s, D13s, mu2, mask2,
        f"W3_AL_step{step_name}_mu_slice3D",
        title=rf"Student $\mu$ diag surface ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )

    plot_slice_2d_heat(D12s, D13s, err_diag, mask2,
        f"W3_AL_step{step_name}_abserr_slice2D",
        title=rf"Absolute error diag slice ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=None
    )
    plot_slice_3d_surface(D12s, D13s, err_diag, mask2,
        f"W3_AL_step{step_name}_abserr_slice3D",
        title=rf"Absolute error diag surface ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=None
    )

    normSig = mpl.colors.Normalize(vmin=float(np.nanmin(sig2[mask2])), vmax=float(np.nanmax(sig2[mask2])))
    plot_slice_2d_heat(D12s, D13s, sig2, mask2,
        f"W3_AL_step{step_name}_sigma_slice2D",
        title=rf"Student $\sigma$ diag slice ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=r"$\sigma^{\mathrm{Pred}}$", cmap="plasma", norm=normSig
    )
    plot_slice_3d_surface(D12s, D13s, sig2, mask2,
        f"W3_AL_step{step_name}_sigma_slice3D",
        title=rf"Student $\sigma$ diag surface ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}",
        clabel=r"$\sigma^{\mathrm{Pred}}$", cmap="plasma", norm=normSig
    )

    plot_face_2d_heat(d13f, d23f, mu_face, maskf,
        f"W3_AL_step{step_name}_mu_face_d12min_slice2D",
        title=rf"Student $\mu$ face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )
    plot_face_3d_surface(D13f, D23f, mu_face, maskf,
        f"W3_AL_step{step_name}_mu_face_d12min_slice3D",
        title=rf"Student $\mu$ face surface ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_C, cmap="viridis", norm=normU
    )

    plot_face_2d_heat(d13f, d23f, err_face, maskf,
        f"W3_AL_step{step_name}_abserr_face_d12min_slice2D",
        title=rf"Absolute error face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=None
    )
    plot_face_3d_surface(D13f, D23f, err_face, maskf,
        f"W3_AL_step{step_name}_abserr_face_d12min_slice3D",
        title=rf"Absolute error face surface ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}",
        clabel=LAB_ABSERR, cmap=ABSERR_CMAP, norm=None
    )

    W2sum_diag = W2sum_pairs(D12s, D13s, D23s, p2)
    dW3_diag_ex = mu2 - (W2sum_diag - W2_anchor)
    normDW3_diag = _norm_from_masked(dW3_diag_ex, mask2)
    plot_slice_2d_heat(D12s, D13s, dW3_diag_ex, mask2,
        f"DeltaW3_AL_step{step_name}_extracted_DeltaW3Pred_diag_slice2D",
        title=(rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ diag ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}" if USE_LATEX else rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ diag ($d_{{23}}=d_{{13}}$), step {step_name}, {GRAFT}"),
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_diag
    )
    plot_slice_3d_surface(D12s, D13s, dW3_diag_ex, mask2,
        f"DeltaW3_AL_step{step_name}_extracted_DeltaW3Pred_diag_slice3D",
        title=(rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ diag surface, step {step_name}, {GRAFT}" if USE_LATEX else rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ diag surface, step {step_name}, {GRAFT}"),
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_diag
    )

    W2sum_face = W2sum_pairs(D12f, D13f, D23f, p2)
    dW3_face_ex = mu_face - (W2sum_face - W2_anchor)
    normDW3_face = _norm_from_masked(dW3_face_ex, maskf)
    plot_face_2d_heat(d13f, d23f, dW3_face_ex, maskf,
        f"DeltaW3_AL_step{step_name}_extracted_DeltaW3Pred_face_d12min_slice2D",
        title=(rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}" if USE_LATEX else rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ face ($d_{{12}}={D12_MIN:.2f}$), step {step_name}, {GRAFT}"),
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_face
    )
    plot_face_3d_surface(D13f, D23f, dW3_face_ex, maskf,
        f"DeltaW3_AL_step{step_name}_extracted_DeltaW3Pred_face_d12min_slice3D",
        title=(rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ face surface, step {step_name}, {GRAFT}" if USE_LATEX else rf"Student extracted $\mathsf{{\Delta}} W_3^{{\mathrm{{Pred}}}}$ face surface, step {step_name}, {GRAFT}"),
        clabel=LAB_DW3, cmap="viridis", norm=normDW3_face
    )

    if (X_tr is not None) and (y_tr is not None) and (X_val is not None) and (y_val is not None):
        plot_parity_each_step(model, X_tr, y_tr, X_val, y_val, ymu, ysd, step_name, target_label, train_set_label=train_set_label)
    if (rmse_tr_hist is not None) and (rmse_val_hist is not None):
        plot_rmse_history_each_step(rmse_tr_hist, rmse_val_hist, step_name, target_label, train_set_label=train_set_label)
    if mst_hist is not None:
        plot_mst_history_each_step(mst_hist, step_name)

# =============================================================================
# Small redundancy removal helpers (core fix lives here)
# =============================================================================
def make_train_xy_from_nodes(train_nodes, y_phys, Xs_feas, feas_pos):
    train_nodes = np.asarray(train_nodes, dtype=int)
    pos = feas_pos[train_nodes]
    m = pos >= 0
    if not np.any(m):
        return np.zeros((0,3)), np.zeros((0,)), np.zeros((0,)), 0.0, 1.0
    X_tr = Xs_feas[pos[m]]
    y_tr = np.asarray(y_phys, float)[m]

    X_tr, y_tr = _sanitize_training_data(X_tr, y_tr)
    if X_tr.shape[0] == 0:
        return X_tr, y_tr, y_tr, 0.0, 1.0

    ymu, ysd = y_fit(y_tr)
    z_tr = y2z(y_tr, ymu, ysd)

    assert X_tr.shape[0] == y_tr.shape[0] == z_tr.shape[0]
    return X_tr, y_tr, z_tr, ymu, ysd

def eval_model_rmse(model, X, y_true_phys, ymu_model, ysd_model):
    mu_z, _ = predict_mu_std_z(model, X)
    yhat = z2y(mu_z, ymu_model, ysd_model)
    return rmse(y_true_phys, yhat)

# =============================================================================
# MAIN
# =============================================================================
def main():
    global FIG_DIR

    args = parse_cli()
    if ALANINE_MODE:_ala_enable(args)
    START_MODE   = args.start_mode
    TARGET_MODE  = args.target_mode
    SYMM_MODE    = args.symmetry_mode
    KMODE        = args.kernel_mode
    LABEL_SOURCE = args.label_source

    USE_LSQR = LABEL_SOURCE in ("lsqr", "pathint")
    TRAIN_SET_LABEL = "queried" if LABEL_SOURCE == "oracle_queried" else "paid"

    TARGET_LABEL = "F" if ALANINE_MODE else ("W3" if TARGET_MODE == "umb" else "Residual")
    print(f"[CONFIG] start_mode={START_MODE}  target_mode={TARGET_MODE}  sym={SYMM_MODE}  "
          f"kernel={KMODE}  label_source={LABEL_SOURCE}  graft={GRAFT}  N3={N3}")
    if LABEL_SOURCE == "lsqr":
        print(f"[CONFIG] LSQR integration={LSQR_INTEGRATION_MODE} "
              f"quad={LSQR_EDGE_QUAD_ORDER} fd_eps={LSQR_FD_EPS} "
              f"edge_batch={LSQR_EDGE_BATCH_SIZE}")
    elif LABEL_SOURCE == "pathint":
        if ALANINE_MODE:print("[CONFIG] PATHINT: paid-node MD endpoint trapezoids; fixed-CV constrained gradients")
        else:print(f"[CONFIG] PATHINT integration=continuous quad={PATHINT_QUAD_ORDER} fd_eps={PATHINT_FD_EPS}")

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    np.set_printoptions(precision=4, suppress=True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    gpflow.config.set_default_float(np.float64)
    gpflow.config.set_default_jitter(JITTER_DEF)

    FIG_DIR = Path(
        f"./figs_{RUN_TAG}_stage2_AL_md_poly_oracle_3D_forceint_"
        f"{GRAFT}_"
        f"edges{INIT_EDGE_COUNT}_face{INIT_FACE_DIAG_COUNT}_space{INIT_SPACE_DIAG_COUNT}_"
        f"{START_MODE}_{TARGET_MODE}_sym{SYMM_MODE}_k{KMODE}_lab{LABEL_SOURCE}"
    )
    if ALANINE_MODE:FIG_DIR=_ALA.root/FIG_DIR.name
    FIG_DIR.mkdir(exist_ok=True)
    print(f"[OUT] FIG_DIR = {FIG_DIR}")
    if SAVE_FIGURE_DATA:
        print(f"[OUT] FIG_DATA_DIR = {FIG_DIR / FIG_DATA_SUBDIR}")

    rng = np.random.default_rng(SEED)
    p2 = PARAMS[GRAFT]["dimer"]
    p3 = PARAMS[GRAFT]["trimer"]

    d12 = np.array([0.]) if ALANINE_MODE else np.linspace(D12_MIN,D12_MAX,N3)
    d13 = np.linspace(D13_MIN,D13_MAX,N3,endpoint=not ALANINE_MODE)
    d23 = np.linspace(D23_MIN,D23_MAX,N3,endpoint=not ALANINE_MODE)
    n12, n13, n23 = len(d12), len(d13), len(d23)

    D12v, D13v, D23v = np.meshgrid(d12, d13, d23, indexing="ij")
    feas = triangle_mask(D12v, D13v, D23v)

    Umb3_raw, W3sw3, W3raw3 = Umb(D12v, D13v, D23v, p2, p3)
    Umb3   = np.where(feas, Umb3_raw, np.nan)
    W3sw3  = np.where(feas, W3sw3,    np.nan)

    feas_flat = np.isfinite(Umb3.ravel(order="C"))
    feasible_g = np.where(feas_flat)[0]
    if feasible_g.size == 0:
        raise RuntimeError("No feasible triangles on this grid. Adjust ranges or mask.")

    P_full = np.c_[D13v.ravel(order="C"), D23v.ravel(order="C"), D12v.ravel(order="C")]

    edge_all = cube_edge_nodes(feasible_g, n12, n13, n23)
    anchor_target = np.r_[_ALA.anchor_rad,0.] if ALANINE_MODE else np.array([D13_MAX,D23_MAX,D12_MAX],float)
    if edge_all.size > 0:
        kdt_edge = cKDTree(P_full[edge_all])
        anchor_g = int(edge_all[int(kdt_edge.query(anchor_target, k=1)[1])])
    else:
        kdt_feas = cKDTree(P_full[feasible_g])
        anchor_g = int(feasible_g[int(kdt_feas.query(anchor_target, k=1)[1])])

    if ALANINE_MODE:_ALA.bind(P_full,anchor_g,n12,n13,n23)

    step0_nodes, step0_extra_edges = _build_step0_geometry_from_counts(
        anchor_g=anchor_g, n12=n12, n13=n13, n23=n23,
        feas_mask_flat=feas_flat,
        edge_count=INIT_EDGE_COUNT,
        face_diag_count=INIT_FACE_DIAG_COUNT,
        space_diag_count=INIT_SPACE_DIAG_COUNT
    )
    step0_nodes_connected = enforce_step0_connected(step0_nodes, anchor_g, P_full, n12, n13, n23, feas_flat)

    print(f"[STEP0 geometry] requested edges/face/space = {INIT_EDGE_COUNT}/{INIT_FACE_DIAG_COUNT}/{INIT_SPACE_DIAG_COUNT}")
    print(f"[STEP0 geometry] nodes(before connect)={len(step0_nodes)} nodes(after connect)={len(step0_nodes_connected)} "
          f"extra_edges={len(step0_extra_edges)} anchor_g={anchor_g}")

    step0_overlay_edges = step0_extra_edges if (step0_extra_edges is not None and len(step0_extra_edges) > 0) else None

    U_raw_flat = Umb3.ravel(order="C")
    U_anchor_raw = float(U_raw_flat[anchor_g])
    U_ref = U_raw_flat - U_anchor_raw

    d13_a, d23_a, d12_a = P_full[anchor_g]
    W2_anchor = float(W2sum_pairs(d12_a, d13_a, d23_a, p2))

    d13_flat = P_full[:, 0]
    d23_flat = P_full[:, 1]
    d12_flat = P_full[:, 2]
    W2sum_flat = W2sum_pairs(d12_flat, d13_flat, d23_flat, p2)
    W2anch_flat = W2sum_flat - W2_anchor
    RES_ref = U_ref - W2anch_flat

    plot_g = feasible_g
    if plot_g.size > PLOT_MAX:
        plot_g = rng.choice(plot_g, size=PLOT_MAX, replace=False)

    normU = _norm_from_clim()
    d12s, d13s, D12s, D13s, D23s, mask2 = build_diag_slice_grid()
    d13f, d23f, D13f, D23f, D12f, maskf = build_face_slice_grid()

    plot_scatter3d_color(
        P_full[plot_g], U_ref[plot_g], LAB_W3_TRUE,
        "W3_01_oracle_W3True_scatter",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ (anchored), {GRAFT}",
        cmap="viridis", norm=normU,
        cbar_ticks=np.arange(CLIM_MIN, CLIM_MAX + 1e-9, 10) if USE_FIXED_CLIM else None,
        scatter_swap_xy=False
    )

    U2, _, _ = oracle_on_slice(D12s, D13s, p2, p3)
    U2a = U2 - U_anchor_raw
    plot_slice_2d_heat(D12s, D13s, U2a, mask2, "W3_01_oracle_W3True_slice2D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ diag slice (anchored), {GRAFT}",
        clabel=LAB_W3_TRUE, cmap="viridis", norm=normU
    )
    plot_slice_3d_surface(D12s, D13s, U2a, mask2, "W3_01_oracle_W3True_slice3D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ diag surface (anchored), {GRAFT}",
        clabel=LAB_W3_TRUE, cmap="viridis", norm=normU
    )

    Uface, _, _ = oracle_on_face(D13f, D23f, D12f, p2, p3)
    Uface_a = Uface - U_anchor_raw
    plot_face_2d_heat(d13f, d23f, Uface_a, maskf, "W3_01_oracle_W3True_face_d12min_slice2D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ face ($d_{{12}}={D12_MIN:.2f}$) (anchored), {GRAFT}",
        clabel=LAB_W3_TRUE, cmap="viridis", norm=normU
    )
    plot_face_3d_surface(D13f, D23f, Uface_a, maskf, "W3_01_oracle_W3True_face_d12min_slice3D",
        title=rf"MD-poly oracle $U_{{\mathrm{{mb}}}}$ face surface ($d_{{12}}={D12_MIN:.2f}$) (anchored), {GRAFT}",
        clabel=LAB_W3_TRUE, cmap="viridis", norm=normU
    )

    dW3_flat = W3sw3.ravel(order="C")
    plot_scatter3d_color(P_full[plot_g], dW3_flat[plot_g], LAB_DW3_TRUE,
        "DeltaW3_02_oracle_DeltaW3True_scatter",
        title=(rf"MD-poly oracle $\mathsf{{\Delta}} W_3^{{\mathrm{{True}}}}$, {GRAFT}" if USE_LATEX
               else rf"MD-poly oracle $\mathsf{{\Delta}} W_3^{{\mathrm{{True}}}}$, {GRAFT}"),
        cmap="viridis", norm=None, scatter_swap_xy=False
    )

    U_full_map = None
    Fx = Fy = Fz = None

    def teacher_query_force_at_nodes(_nodes):
        return

    if USE_LSQR:
        Ugrid = U_ref.reshape(n12, n13, n23, order="C")
        Ugrid_nan = np.where(feas, Ugrid, np.nan)

        dU_dd12, dU_dd13, dU_dd23 = compute_teacher_forces(Ugrid_nan, feas, d12, d13, d23)
        Fd12_flat = dU_dd12.ravel(order="C")
        Fd13_flat = dU_dd13.ravel(order="C")
        Fd23_flat = dU_dd23.ravel(order="C")

        Fx = np.full(P_full.shape[0], np.nan, float)
        Fy = np.full(P_full.shape[0], np.nan, float)
        Fz = np.full(P_full.shape[0], np.nan, float)

        def teacher_query_force_at_nodes(nodes):
            nodes = np.array(sorted(set(map(int, nodes))), dtype=int)
            if nodes.size == 0:
                return
            nodes = nodes[feas_flat[nodes]]
            if nodes.size == 0:
                return
            if ALANINE_MODE:
                gradients=_ALA.query(nodes)
                Fx[nodes]=gradients[:,0];Fy[nodes]=gradients[:,1];Fz[nodes]=0.
                return
            Fx[nodes] = Fd13_flat[nodes]
            Fy[nodes] = Fd23_flat[nodes]
            Fz[nodes] = Fd12_flat[nodes]

        if DO_FULLGRID_LSQR_DIAGNOSTIC:
            teacher_query_force_at_nodes(feasible_g)
            U_full, nodes_full = lsqr_U_on_known_nodes_3d(
                known_nodes=feasible_g, anchor_g=anchor_g, P_full=P_full,
                Fx=Fx, Fy=Fy, Fz=Fz,
                n12=n12, n13=n13, n23=n23, feas_mask_flat=feas_flat,
                iter_lim=20000,
                graph_mode=LSQR_GRAPH_MODE,
                knn_k=LSQR_KNN_K,
                knn_max_dist=LSQR_KNN_MAX_DIST,
                extra_edges=None,
                integration_mode=LSQR_INTEGRATION_MODE,
                p2=p2,
                p3=p3,
                quad_order=LSQR_EDGE_QUAD_ORDER,
                fd_eps=LSQR_FD_EPS,
                edge_batch_size=LSQR_EDGE_BATCH_SIZE
            )
            U_full_map = np.full(P_full.shape[0], np.nan, float)
            U_full_map[nodes_full] = U_full
            keep = np.isfinite(U_full_map[feasible_g]) & np.isfinite(U_ref[feasible_g])
            print(f"[LSQR full-grid diagnostic] RMSE vs oracle (feasible) = "
                  f"{rmse(U_ref[feasible_g][keep], U_full_map[feasible_g][keep]):.6f}")
            if PLOT_FULLGRID_LSQR_DIAGNOSTIC:
                plot_scatter3d_color(
                    P_full[plot_g], U_full_map[plot_g], LAB_C,
                    "Umb_03_diag_LSQR_fullgrid_scatter",
                    title=rf"LSQR PMF from oracle forces (full-grid diagnostic), {GRAFT}",
                    cmap="viridis", norm=normU, scatter_swap_xy=False
                )
            Fx[:] = np.nan; Fy[:] = np.nan; Fz[:] = np.nan

    P_feas = P_full[feasible_g]
    P_feat_feas = _features_from_P(P_feas, symmetry_mode=SYMM_MODE)
    xmu, xsd = standardize_fit(P_feat_feas)
    Xs_feas = standardize_apply(P_feat_feas, xmu, xsd)

    W2sum_feas = W2sum_pairs(P_feas[:, 2], P_feas[:, 0], P_feas[:, 1], p2)
    W2anch_feas = W2sum_feas - W2_anchor

    feas_pos = -np.ones(P_full.shape[0], dtype=int)
    feas_pos[feasible_g] = np.arange(feasible_g.size, dtype=int)

    exclude0 = np.union1d(step0_nodes_connected, np.array([anchor_g], dtype=int))
    val_pool = np.setdiff1d(feasible_g, exclude0)
    n_val = int(np.clip(VAL_FRAC * feasible_g.size, VAL_MIN, VAL_MAX))
    if val_pool.size == 0:
        val_pool = feasible_g[feasible_g != anchor_g]
    val_g = rng.choice(val_pool, size=min(n_val, val_pool.size), replace=False)

    is_val = np.zeros(P_full.shape[0], dtype=bool)
    is_val[val_g] = True
    is_val[int(anchor_g)] = False

    allowed_train_mask = feas_flat & (~is_val)
    allowed_train_mask[int(anchor_g)] = True

    y_val_true = U_ref[val_g] if TARGET_MODE == "umb" else RES_ref[val_g]
    X_val = Xs_feas[feas_pos[val_g]]
    if ALANINE_MODE:
        val_g=np.array([],dtype=int)
        is_val[:]=False
        allowed_train_mask=feas_flat.copy()
        X_val,y_val_true=_ALA.validation(n_val,xmu,xsd,U_anchor_raw)

    def _oracle_target_on_nodes(nodes):
        nodes = np.asarray(nodes, dtype=int)
        if TARGET_MODE == "umb":
            return np.asarray(U_ref[nodes], float)
        else:
            return np.asarray(RES_ref[nodes], float)

    def _build_training_nodes_and_labels(label_source, train_all, train_q,
                                         anchor_g, P_full, n12, n13, n23, feas_mask_flat,
                                         Fx, Fy, Fz, graph_mode, extra_edges):
        ls = str(label_source).lower()

        if ls == "pathint":
            U_paid, paid_nodes = path_integrated_U_on_known_nodes_3d(
                known_nodes=np.array(sorted(set(map(int, train_all))), dtype=int),
                anchor_g=anchor_g,
                P_full=P_full,
                n12=n12,
                n13=n13,
                n23=n23,
                feas_mask_flat=feas_mask_flat,
                p2=p2,
                p3=p3,
                graph_mode=graph_mode,
                knn_k=LSQR_KNN_K,
                knn_max_dist=LSQR_KNN_MAX_DIST,
                extra_edges=extra_edges,
                quad_order=PATHINT_QUAD_ORDER,
                fd_eps=PATHINT_FD_EPS,
            )
            ok = np.isfinite(U_paid)
            paid_nodes = paid_nodes[ok]
            U_paid = U_paid[ok]
            if paid_nodes.size == 0:
                return np.array([], int), np.array([], float), None, None
            if paid_nodes.size > 0:
                u_true = np.asarray(U_ref[paid_nodes], float)
                err = U_paid - u_true
                bias = float(np.mean(err))
                print(
                    ("[PATHINT vs external diagnostic] " if ALANINE_MODE else "[PATHINT vs ORACLE paid] ") +
                    f"n={paid_nodes.size} "
                    f"RMSE={np.sqrt(np.mean(err**2)):.6e} "
                    f"MAE={np.mean(np.abs(err)):.6e} "
                    f"bias={bias:.6e} "
                    f"RMSE_bias_removed={np.sqrt(np.mean((err-bias)**2)):.6e} "
                    f"maxabs={np.max(np.abs(err)):.6e}"
                )
            y = U_paid.copy()
            if TARGET_MODE == "residual":
                y = y - W2anch_flat[paid_nodes]
            return paid_nodes, y, paid_nodes, U_paid

        if ls == "lsqr":
            U_paid, paid_nodes = lsqr_U_on_known_nodes_3d(
                known_nodes=train_all, anchor_g=anchor_g, P_full=P_full,
                Fx=Fx, Fy=Fy, Fz=Fz,
                n12=n12, n13=n13, n23=n23, feas_mask_flat=feas_mask_flat,
                iter_lim=12000,
                graph_mode=graph_mode,
                knn_k=LSQR_KNN_K,
                knn_max_dist=LSQR_KNN_MAX_DIST,
                extra_edges=extra_edges,
                integration_mode=LSQR_INTEGRATION_MODE,
                p2=p2,
                p3=p3,
                quad_order=LSQR_EDGE_QUAD_ORDER,
                fd_eps=LSQR_FD_EPS,
                edge_batch_size=LSQR_EDGE_BATCH_SIZE
            )
            ok = np.isfinite(U_paid)
            paid_nodes = paid_nodes[ok]
            U_paid = U_paid[ok]
            if paid_nodes.size == 0:
                return np.array([], int), np.array([], float), None, None

            y = U_paid.copy()
            if TARGET_MODE == "residual":
                y = y - W2anch_flat[paid_nodes]
            return paid_nodes, y, paid_nodes, U_paid

        elif ls == "oracle_paid":
            nodes = np.array(sorted(set(map(int, train_all))), dtype=int)
            nodes = nodes[feas_mask_flat[nodes]]
            y = _oracle_target_on_nodes(nodes)
            return nodes, y, None, None

        elif ls == "oracle_queried":
            nodes = np.array(sorted(set(map(int, train_q))), dtype=int)
            nodes = nodes[feas_mask_flat[nodes]]
            y = _oracle_target_on_nodes(nodes)
            return nodes, y, None, None

        else:
            raise ValueError(f"Unknown label_source={label_source}")

    train_q   = set(map(int, step0_nodes_connected.tolist()))
    train_all = set(train_q)

    extra0 = step0_extra_edges if (step0_extra_edges is not None and len(step0_extra_edges) > 0) else None
    if USE_LSQR:
        teacher_query_force_at_nodes(step0_nodes_connected)

    train_nodes, y_phys_raw, paid_nodes_lsqr, U_paid_lsqr = _build_training_nodes_and_labels(
        LABEL_SOURCE, train_all, train_q,
        anchor_g, P_full, n12, n13, n23, feas_flat,
        Fx, Fy, Fz,
        graph_mode=STEP0_LSQR_GRAPH_MODE,
        extra_edges=extra0
    )
    if train_nodes.size == 0:
        raise RuntimeError("Initial labeling produced no training labels (check LSQR/forces or node sets).")

    X_tr, y_tr, z_tr, ymu, ysd = make_train_xy_from_nodes(train_nodes, y_phys_raw, Xs_feas, feas_pos)
    if X_tr.shape[0] < 2:
        raise RuntimeError("Not enough training points after sanitization.")

    prev_m = build_gpr(
        X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, INIT_ITERS,
        m_prev=None
    )
    prev_ymu, prev_ysd = ymu, ysd

    rtr0 = eval_model_rmse(prev_m, X_tr, y_tr, prev_ymu, prev_ysd)
    rv0  = eval_model_rmse(prev_m, X_val, y_val_true, prev_ymu, prev_ysd)

    rmse_tr_hist  = [rtr0]
    rmse_val_hist = [rv0]
    mst_hist      = [mst_total_length(P_full, np.array(sorted(train_all), int))]

    print(f"[AL init] target={TARGET_MODE}  paid={len(train_all)} queried={len(train_q)}  "
          f"RMSE(train-{TRAIN_SET_LABEL},{TARGET_LABEL})={rtr0:.4f}  RMSE(val-oracle,{TARGET_LABEL})={rv0:.4f}")

    assert len(set(val_g.tolist()) & train_all) == 0
    assert len(set(val_g.tolist()) & train_q) == 0

    pending_plot = {
        "step_name": "00",
        "model": prev_m,
        "ymu": prev_ymu,
        "ysd": prev_ysd,
        "train_all": set(train_all),
        "train_q": set(train_q),
        "X_tr": X_tr,
        "y_tr": y_tr.copy(),
        "X_val": X_val,
        "y_val": y_val_true,
        "step0_extra_edges": extra0,
        "paid_nodes": (paid_nodes_lsqr.copy() if paid_nodes_lsqr is not None else None),
        "U_paid": (U_paid_lsqr.copy() if U_paid_lsqr is not None else None),
    }

    subN = min(6000, Xs_feas.shape[0])
    if subN >= 2:
        sub_idx = rng.choice(np.arange(Xs_feas.shape[0]), size=subN, replace=False) if subN < Xs_feas.shape[0] else np.arange(subN)
        tree_sub = cKDTree(Xs_feas[sub_idx])
        dnn, _ = tree_sub.query(Xs_feas[sub_idx], k=2)
        base_h = float(np.median(dnn[:, 1])) if dnn.ndim == 2 and dnn.shape[1] >= 2 else 0.15
    else:
        base_h = 0.15
    base_h = max(base_h, 0.10)
    min_sep = MIN_SEP_MULT * base_h

    for it in range(AL_ITERS):
        mask_unl = np.ones(P_full.shape[0], bool)
        if len(train_all) > 0:
            mask_unl[list(train_all)] = False
        if val_g.size > 0:
            mask_unl[val_g] = False

        cand_g_all = feasible_g[mask_unl[feasible_g]]
        if cand_g_all.size == 0:
            print(f"[AL {it+1:02d}] stop: no candidates left.")
            break

        _, std_old_z = predict_mu_std_z(prev_m, Xs_feas)

        tr_nodes_all = np.array(sorted(train_all), dtype=int)
        tr_pos_all = feas_pos[tr_nodes_all]
        tr_pos_all = tr_pos_all[tr_pos_all >= 0]
        if tr_pos_all.size == 0:
            tr_pos_all = np.array([feas_pos[anchor_g]], int)

        kd = cKDTree(Xs_feas[tr_pos_all])
        cand_pos_all = feas_pos[cand_g_all]
        d_to_train_all, _ = kd.query(Xs_feas[cand_pos_all], k=1)

        gate = d_to_train_all >= min_sep
        cand_g   = cand_g_all[gate]
        cand_pos = cand_pos_all[gate]
        d_to_train = d_to_train_all[gate]

        if cand_g.size < max(1, int(POOL_MIN_FRACTION * feasible_g.size)):
            old = min_sep
            min_sep *= ADAPT_MINSEP_DECAY
            print(f"[AL {it+1:02d}] relax min_sep: {old:.4g} -> {min_sep:.4g} (pool {cand_g.size})")
            gate = d_to_train_all >= min_sep
            cand_g   = cand_g_all[gate]
            cand_pos = cand_pos_all[gate]
            d_to_train = d_to_train_all[gate]

        if cand_g.size == 0:
            cand_g   = cand_g_all
            cand_pos = cand_pos_all
            d_to_train = d_to_train_all

        std_c = std_old_z[cand_pos]
        noise_var = float(prev_m.likelihood.variance.numpy())

        info = 0.5 * np.log1p((std_c**2) / (noise_var + 1e-12))
        spacing = np.minimum(d_to_train, 4.0 * base_h) ** 0.5
        score = info * spacing
        score = np.nan_to_num(score, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        print(f"Score stats: min={score.min():.3f}, max={score.max():.3f}, std={score.std():.3f}")

        order = np.argsort(-score)
        if order.size == 0 or (not np.isfinite(score).any()):
            order = np.argsort(-std_c)

        take_g = []
        taken = []
        r_min = base_h
        for idx in order:
            g = int(cand_g[int(idx)])
            p = Xs_feas[int(cand_pos[int(idx)])]
            if not taken:
                take_g.append(g); taken.append(p)
            else:
                dmin_now = np.min(np.linalg.norm(p - np.vstack(taken), axis=1))
                if dmin_now >= r_min:
                    take_g.append(g); taken.append(p)
            if len(take_g) >= AL_BATCH:
                break
        if len(take_g) == 0:
            take_g = [int(x) for x in cand_g[order[:AL_BATCH]]]
        take_g = np.array(take_g, dtype=int)

        score_map = {int(cand_g[i]): float(score[i]) for i in range(cand_g.size)}

        cand_overlay = set()
        if SHOW_CAND_OVERLAY and cand_g.size > 0:
            tree_c = cKDTree(Xs_feas[cand_pos])
            k_use = min(int(KNN_K), cand_g.size)
            for g in take_g:
                _, nn_loc = tree_c.query(Xs_feas[feas_pos[g]], k=k_use)
                nn_loc = np.atleast_1d(nn_loc)
                for kk in nn_loc:
                    cand_overlay.add(int(cand_g[int(kk)]))

        cand_overlay_idx = np.array(sorted(cand_overlay), dtype=int)
        if cand_overlay_idx.size > CAND_SHOW_N:
            cs = np.asarray([score_map.get(int(g), -np.inf) for g in cand_overlay_idx], float)
            keep = np.argsort(-cs)[:CAND_SHOW_N]
            cand_overlay_idx = cand_overlay_idx[keep]

        plot_all_figs_for_step(
            pending_plot["step_name"], pending_plot["model"],
            plot_g, P_full, P_feas, feas_pos, Xs_feas,
            xmu, xsd, pending_plot["ymu"], pending_plot["ysd"], normU,
            U_ref, U2a, Uface_a,
            pending_plot["train_all"], pending_plot["train_q"], val_g, anchor_g,
            d12s, d13s, D12s, D13s, D23s, mask2,
            d13f, d23f, D13f, D23f, D12f, maskf,
            p2, W2_anchor,
            score_map_for_this_step=score_map,
            cand_overlay_idx_for_this_step=cand_overlay_idx,
            X_tr=pending_plot["X_tr"], y_tr=pending_plot["y_tr"],
            X_val=pending_plot["X_val"], y_val=pending_plot["y_val"],
            rmse_tr_hist=rmse_tr_hist, rmse_val_hist=rmse_val_hist, mst_hist=mst_hist,
            n12=n12, n13=n13, n23=n23, feas_flat=feas_flat,
            step0_extra_edges=pending_plot.get("step0_extra_edges", None),
            step0_overlay_edges=step0_overlay_edges,
            target_mode=TARGET_MODE,
            W2anch_feas=W2anch_feas,
            target_label=TARGET_LABEL,
            symmetry_mode=SYMM_MODE,
            train_set_label=TRAIN_SET_LABEL,
            U_full_map=U_full_map,
            paid_nodes_step=pending_plot.get("paid_nodes", None),
            U_paid_step=pending_plot.get("U_paid", None)
        )

        for g in take_g.tolist():
            train_q.add(int(g))

        train_set_all = set(train_all)
        pending = [int(g) for g in take_g.tolist()]
        newly_paid = []

        while pending:
            cur_nodes = np.array(sorted(train_set_all), dtype=int)
            if cur_nodes.size == 0:
                cur_nodes = np.array([int(anchor_g)], dtype=int)
                train_set_all.add(int(anchor_g))

            kd_cur = _ala_tree(P_full[cur_nodes]) if ALANINE_MODE else cKDTree(P_full[cur_nodes])
            nearest_points=P_full[pending,:2]+np.pi if ALANINE_MODE else P_full[pending]
            dists, locs = kd_cur.query(nearest_points,k=1)
            best_pos = int(np.argmin(np.atleast_1d(dists)))
            g_new = int(pending[best_pos])
            g_tgt = int(cur_nodes[int(np.atleast_1d(locs)[best_pos])])

            added_path = shortest_path_add_3d(
                train_set_all, g_new, g_tgt,
                P_full, n12, n13, n23, allowed_train_mask
            )
            newly_paid.extend(added_path)
            pending.pop(best_pos)

        newly_paid = [int(g) for g in newly_paid if allowed_train_mask[int(g)]]
        if USE_LSQR:
            teacher_query_force_at_nodes(newly_paid)

        train_all = set(train_set_all)

        assert len(set(val_g.tolist()) & train_all) == 0
        assert len(set(val_g.tolist()) & train_q) == 0

        train_nodes, y_phys_raw, paid_nodes_lsqr, U_paid_lsqr = _build_training_nodes_and_labels(
            LABEL_SOURCE, train_all, train_q,
            anchor_g, P_full, n12, n13, n23, feas_flat,
            Fx, Fy, Fz,
            graph_mode=LSQR_GRAPH_MODE,
            extra_edges=None
        )
        if train_nodes.size == 0:
            print(f"[AL {it+1:02d}] WARNING: no training labels produced; skipping update.")
            continue

        X_tr, y_tr, z_tr, ymu, ysd = make_train_xy_from_nodes(train_nodes, y_phys_raw, Xs_feas, feas_pos)
        if X_tr.shape[0] < 2:
            print(f"[AL {it+1:02d}] WARNING: too few points after sanitization; skipping update.")
            continue

        rtr_prev = eval_model_rmse(prev_m, X_tr, y_tr, prev_ymu, prev_ysd)
        rv_prev  = eval_model_rmse(prev_m, X_val, y_val_true, prev_ymu, prev_ysd)

        mprev_for_build = prev_m if (START_MODE == "warm") else None

        try:
            m_new = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, 800, m_prev=mprev_for_build)
        except Exception as e:
            print(f"[AL {it+1:02d}] WARN: GP build failed ({e}); retrying with safer settings.")
            gpflow.config.set_default_jitter(1e-2)
            m_new = build_gpr(X_tr, z_tr, KMODE, KERNEL_SPEC, ARD, 400, m_prev=mprev_for_build)

        rtr_new = eval_model_rmse(m_new, X_tr, y_tr, ymu, ysd)
        rv_new  = eval_model_rmse(m_new, X_val, y_val_true, ymu, ysd)

        mu_new_tr_z, std_new_tr_z = predict_mu_std_z(m_new, X_tr)
        ok_new = (np.all(np.isfinite(mu_new_tr_z)) and
                  np.all(np.isfinite(std_new_tr_z)) and
                  np.all(std_new_tr_z > 0) and
                  np.isfinite(rtr_new))

        if ok_new and (rtr_new <= rtr_prev * (1.0 + ACCEPT_TOL)):
            prev_m = m_new
            prev_ymu, prev_ysd = ymu, ysd
            rtr_acc, rv_acc = rtr_new, rv_new
            acc_flag = "ACCEPT"
        else:
            rtr_acc, rv_acc = rtr_prev, rv_prev
            acc_flag = "REJECT"

        rmse_tr_hist.append(rtr_acc)
        rmse_val_hist.append(rv_acc)
        mst_hist.append(mst_total_length(P_full, np.array(sorted(train_all), int)))

        step_next = f"{it+1:02d}"
        print(f"[AL {step_next}] {acc_flag}  target={TARGET_MODE}  paid={len(train_all)} "
              f"queried={len(train_q)}  RMSE(train-{TRAIN_SET_LABEL},{TARGET_LABEL})={rtr_acc:.4f}  RMSE(val-oracle,{TARGET_LABEL})={rv_acc:.4f}")

        pending_plot = {
            "step_name": step_next,
            "model": prev_m,
            "ymu": prev_ymu,
            "ysd": prev_ysd,
            "train_all": set(train_all),
            "train_q": set(train_q),
            "X_tr": X_tr,
            "y_tr": y_tr.copy(),
            "X_val": X_val,
            "y_val": y_val_true,
            "step0_extra_edges": None,
            "paid_nodes": (paid_nodes_lsqr.copy() if paid_nodes_lsqr is not None else None),
            "U_paid": (U_paid_lsqr.copy() if U_paid_lsqr is not None else None),
        }

    plot_all_figs_for_step(
        pending_plot["step_name"], pending_plot["model"],
        plot_g, P_full, P_feas, feas_pos, Xs_feas,
        xmu, xsd, pending_plot["ymu"], pending_plot["ysd"], normU,
        U_ref, U2a, Uface_a,
        pending_plot["train_all"], pending_plot["train_q"], val_g, anchor_g,
        d12s, d13s, D12s, D13s, D23s, mask2,
        d13f, d23f, D13f, D23f, D12f, maskf,
        p2, W2_anchor,
        score_map_for_this_step=None,
        cand_overlay_idx_for_this_step=np.array([], dtype=int),
        X_tr=pending_plot["X_tr"], y_tr=pending_plot["y_tr"],
        X_val=pending_plot["X_val"], y_val=pending_plot["y_val"],
        rmse_tr_hist=rmse_tr_hist, rmse_val_hist=rmse_val_hist, mst_hist=mst_hist,
        n12=n12, n13=n13, n23=n23, feas_flat=feas_flat,
        step0_extra_edges=None,
        step0_overlay_edges=step0_overlay_edges,
        target_mode=TARGET_MODE,
        W2anch_feas=W2anch_feas,
        target_label=TARGET_LABEL,
        symmetry_mode=SYMM_MODE,
        train_set_label=TRAIN_SET_LABEL,
        U_full_map=U_full_map,
        paid_nodes_step=pending_plot.get("paid_nodes", None),
        U_paid_step=pending_plot.get("U_paid", None)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    its = np.arange(1, len(rmse_tr_hist) + 1)
    ax.plot(its, rmse_tr_hist, "-", lw=5, ms=80, label=f"Train")
    ax.plot(its, rmse_val_hist, "-", lw=5, ms=80, label=("External FES diagnostic" if ALANINE_MODE else "Val"))
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("RMSE", fontsize=AXIS_LABEL_FZ)
    ax.set_xlim(0.5, len(its) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    ax.legend(frameon=False, fontsize=LEGEND_FZ, loc="upper right",
          markerscale=2.0, handlelength=2.0, handletextpad=0.6,
          labelspacing=0.5, borderpad=0.2, borderaxespad=0.2)
    save_numeric_figure_data(
        "W3_90_rmse_vs_iter",
        np.column_stack([its, np.asarray(rmse_tr_hist, float), np.asarray(rmse_val_hist, float)]),
        "iteration,rmse_train,rmse_val",
    )
    savefig(fig, "W3_90_rmse_vs_iter")

    fig, ax = plt.subplots(figsize=FIGSIZE_RMSE)
    steps = np.arange(len(mst_hist))
    ax.plot(steps, mst_hist, "-o", lw=5, ms=16, mfc="none")
    ax.set_xlabel("AL step", fontsize=AXIS_LABEL_FZ)
    ax.set_ylabel("MST length", fontsize=AXIS_LABEL_FZ)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FZ, pad=6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FZ, pad=6)
    save_numeric_figure_data("W3_91_mst_vs_iter", np.column_stack([steps, np.asarray(mst_hist, float)]), "al_step,mst_length")
    savefig(fig, "W3_91_mst_vs_iter")

    steps_out = np.arange(len(rmse_tr_hist), dtype=int)

    metrics_out = np.column_stack([
        steps_out,
        np.asarray(rmse_tr_hist, float),
        np.asarray(rmse_val_hist, float),
        np.asarray(mst_hist, float),
    ])

    np.savetxt(
        FIG_DIR / "al_metrics.csv",
        metrics_out,
        header="step rmse_train rmse_val mst_length",
        fmt=["%d", "%.10f", "%.10f", "%.10f"],
    )

    plt.close("all")

# Original entry point is dispatched after the appended alanine definitions.



ATTACHED_SOURCE_SHA256 = '482dd2a410e44f307d16986e2e210d595809af3066d7cdd020f94e765dd87e76'
SOURCE_AUDIT = {'original_attachment_sha256': '482dd2a410e44f307d16986e2e210d595809af3066d7cdd020f94e765dd87e76', 'original_function_count': 101, 'unchanged_original_functions': ['_setup_mpl_tex_cache_to_tmp', 'parse_cli', 'switch_poly', 'switch_cos2', 'y_dimer_basis', 'y_trimer_basis', 'W2', 'deltaW3_raw', 'sf_sum', '_safe_output_name', '_figure_data_dir', 'save_numeric_figure_data', 'save_scatter_figure_data', 'save_diag_slice_figure_data', 'save_face_slice_figure_data', 'save_overlay_figure_data', 'savefig', 'standardize_fit', 'standardize_apply', 'y_fit', 'y2z', 'z2y', 'unique_rows_idx', 'rmse', '_sanitize_training_data', '_norm_from_clim', '_norm_from_masked', '_abs_err', 'structured_kernel_1d3d', 'psd_rescue', '_try_minimize', 'warm_start_params', 'build_gpr', 'predict_mu_std_z', '_apply_axis_mapping', '_mapped_labels', '_mapped_limits', '_apply_axis_inversions', '_compact_number_label', '_distance_ticks', '_apply_distance_ticks_3d', '_format_energy_zaxis', '_format_colorbar_ticks', '_adjust_pinned_zlabel_for_ticks', '_apply_distance_ticks_2d', '_make_3d_ticks_visible', '_add_top_colorbar', '_pin_3d_zlabel_2d', 'plot_scatter3d_color', 'plot_slice_2d_heat', 'plot_slice_3d_surface', 'plot_face_2d_heat', 'plot_face_3d_surface', 'gp_predict_on_slice', 'gp_predict_on_face', 'oracle_on_face', 'ijk_to_g', 'g_to_ijk', 'manhattan_walk_add_3d', 'build_edges_from_nodes', 'build_edges_axis1_from_nodes', 'build_edges_knn_from_nodes', 'lsqr_U_on_known_nodes_3d', '_Umb_value_from_P_3d', '_grad_Umb_fd_3d', '_edge_work_quadrature_3d_batch', '_line_nodes_edges_between', '_all_cube_edges_ijk', '_select_connected_cube_edges_ijk', '_all_face_diagonals_ijk', '_all_space_diagonals_ijk', '_rank_lines_by_anchor_endpoint', '_components_in_step0', 'enforce_step0_connected', 'overlay_train_val_mst_factory', 'plot_mst_history_each_step', 'plot_parity_each_step', 'plot_lsqr_paid_vs_fullgrid_parity', 'fill_infeasible_by_nearest', 'make_train_xy_from_nodes', 'eval_model_rmse'], 'changed_original_functions': ['Umb', 'triangle_mask', 'W2sum_pairs', '_features_from_P', 'mst_total_length', 'kernel_from_spec', 'build_diag_slice_grid', 'build_face_slice_grid', 'oracle_on_slice', 'neighbors_26', 'shortest_path_add_3d', '_edge_work_quadrature_3d', 'path_integrated_U_on_known_nodes_3d', 'cube_edge_nodes', '_build_step0_geometry_from_counts', 'mst_edges_over_allowed_edges', 'plot_rmse_history_each_step', 'compute_teacher_forces', 'plot_all_figs_for_step', 'main'], 'core_GP_functions_identical': True, 'controller_settings': {'kernel': 'm32+rq*se', 'start_mode': 'cold', 'al_batch': 5, 'al_iterations': 50}, 'structural_changes': 'Guarded periodic 2-CV mode, MD force labels, off-grid diagnostic points, and 2D alanine plots; new code appended', 'runtime_validation': 'New single-file scheduler/MD/GP integration must be verified on CRC; no local end-to-end CP2K claim.'}
# =============================================================================
# APPENDED ALANINE ADAPTER -- original GP controller remains above this section.
# P_full keeps its three-column storage convention: (phi, psi, constant zero).
# Only phi and psi are CVs; the GP uses their four sine/cosine features.
# =============================================================================
import fcntl as _ala_fcntl
from scipy.interpolate import RegularGridInterpolator as _AlaInterpolator

_ALA = None


def _ala_features(P):
    p=np.asarray(P,float)[...,:2]
    return np.stack((np.cos(p[...,0]),np.sin(p[...,0]),
                     np.cos(p[...,1]),np.sin(p[...,1])),axis=-1)


def _ala_displacement(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    d=np.array(d,copy=True)
    d[...,:2]=wrapped(d[...,:2])
    return d


def _ala_tree(P):
    return cKDTree(np.asarray(P)[:,:2]+np.pi,boxsize=2*np.pi)


def _ala_neighbors(g,n12,n13,n23):
    if n12!=1:raise ValueError('Alanine mode uses two CVs, stored on a 1 x N x N grid')
    _,i,j=g_to_ijk(g,n12,n13,n23)
    return [ijk_to_g(0,(i+a)%n13,(j+b)%n23,n12,n13,n23)
            for a in (-1,0,1) for b in (-1,0,1) if a or b]


def _ala_seed_geometry(anchor,n12,n13,n23):
    # phi=-pi is the same periodic seam as the right box edge phi=+pi.
    # Sample every psi on that seam once; include the closing periodic edge.
    nodes=[ijk_to_g(0,0,k,n12,n13,n23) for k in range(n23)]
    edges=[(int(a),int(b)) for a,b in zip(nodes,nodes[1:]+nodes[:1])]
    return np.array(nodes,int),edges


def _ala_face_grid():
    phi=np.linspace(-np.pi,np.pi,N3,endpoint=False)
    psi=phi.copy();PHI,PSI=np.meshgrid(phi,psi,indexing='xy')
    return phi,psi,PHI,PSI,np.zeros_like(PHI),np.ones_like(PHI,dtype=bool)


def _ala_diag_grid():
    phi,psi,PHI,PSI,Z,mask=_ala_face_grid()
    return psi,phi,PSI,PHI,PSI,mask


def _ala_reference3(d12,phi,psi):
    a,b,c=np.broadcast_arrays(d12,phi,psi)
    p=np.stack((b,c),axis=-1)
    value=_ALA.reference(p)
    return value,np.zeros_like(value),np.zeros_like(value)


def _ala_original_parameters():
    # Preserve the attached code's cold/warm choice, kernel and AL batch defaults.
    return dict(start_mode=DEFAULT_START_MODE,kernel=KERNEL_SPEC,al_batch=AL_BATCH,
                al_iterations=AL_ITERS,GP_initial_iterations=INIT_ITERS,seed=SEED)


class _AlaBudgetStop(RuntimeError):pass


class _SingleAlanineAdapter:
    def __init__(self):
        if _single_backend_signature()!=SCIENTIFIC_CODE_SHA256:
            raise RuntimeError('Scientific code was edited; verify the change before reusing a campaign')
        self.project=Path(SINGLE_PROJECT).resolve()
        self.root=_single_campaign_root()
        self.root.mkdir(exist_ok=True)
        self.lock=_SPYDER_CONTROLLER_LOCK or (self.root/'adapter.lock').open('w')
        try:_ala_fcntl.flock(self.lock,_ala_fcntl.LOCK_EX|_ala_fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another single-file campaign is already running')
        oldroot=self.project/'ALANINE_CP2K_PATHINT_FINAL'
        oldbundle=self.project/'GAPS_CP2K_AUTOMATED_RUN'
        if not oldroot.is_dir():raise FileNotFoundError('The validated CP2K campaign/model is required: '+str(oldroot))
        self.cfg=read(oldroot/'campaign_config.json')
        # Fixed-cost benchmark: ONE independent MD replica per sampled node.
        # Override the in-memory copy, leaving the validated source config untouched.
        self.cfg['replicas_per_window']=1
        self.settings=read(self.project/'alanine_config.json')
        # Retain the attached controller's algorithm settings, not the earlier
        # toy2 controller's different kernel/batch/start-mode defaults.
        self.settings.update(seed=SEED,al_iterations=AL_ITERS,al_batch=AL_BATCH,grid_n=GRID_RESOLUTION,max_paid_nodes=1000,
                             initialization='right periodic seam: phi=-pi, full psi line',dispatch='SGE parallel window array')
        self.kbt=.008314462618*self.cfg['temperature_K']
        self.n=int(self.settings['grid_n'])
        self.anchor_rad=np.array([-np.pi,-np.pi],float)
        self.settings['anchor_rad']=self.anchor_rad.tolist()
        self.gradients={};self.draws={};self.covariances={};self.reference_interpolator=None
        if self.cfg['backend']!='CP2K fixed-CV SHAKE/RATTLE, no harmonic restraints':
            raise ValueError('Expected the validated fixed-CV backend, not the old spring workflow')
        if self.cfg['replicas_per_window']!=1 or self.cfg['temperature_K']!=300:
            raise ValueError('Unexpected sampling settings')
        fft=read(oldbundle/'fft_gate.json')
        if fft['selected_grid']!=self.cfg['fft_grid']:raise ValueError('FFT validation/config mismatch')
        transfer=read(oldbundle/'static_transfer_report.json')
        if not transfer['preliminary_transfer_gate']:raise ValueError('Static-transfer check did not pass')
        self.model=read(oldroot/'cp2k_model/model.json')
        import parmed as _ala_pmd
        topology=_ala_pmd.load_file(str(oldroot/'cp2k_model/alanine.prmtop'))
        self.model['bond_graph']=sorted([sorted([b.atom1.idx,b.atom2.idx]) for b in topology.bonds])
        identity=dict(scientific_code_sha256=SCIENTIFIC_CODE_SHA256,worker=self.cfg,
                      settings=self.settings,model_sha256=hashlib.sha256(json.dumps(self.model,sort_keys=True).encode()).hexdigest(),
                      topology_sha256=digest(oldroot/'cp2k_model/alanine.prmtop'),
                      original_controller_sha256=ATTACHED_SOURCE_SHA256)
        manifest=self.root/'input_manifest.json'
        if manifest.exists():
            previous=read(manifest)
            if not _single_manifest_compatible(previous,identity):
                raise RuntimeError('Single-file source/settings changed; use a different campaign directory')
            if previous!=identity:
                backup=manifest.with_suffix('.json.before_live_display')
                if not backup.exists():shutil.copy2(manifest,backup)
                write(manifest,identity)
        else:
            for source,target in ((oldroot/'cp2k_model',self.root/'cp2k_model'),
                                  (self.project/'simulations_essentials',self.root/'simulations_essentials')):
                shutil.copytree(source,target)
            write(self.root/'cp2k_model/model.json',self.model)
            write(manifest,identity)
        frozen=self.root/'runtime_single.py'
        if not frozen.exists():shutil.copy2(Path(__file__).resolve(),frozen)
        else:
            marker='# COMPLETE JOINT-CV GEOMETRY AND MEAN-FORCE CODE\n'
            frozen_digest=_single_scientific_signature(frozen.read_text())
            if frozen_digest!=SCIENTIFIC_CODE_SHA256:
                if frozen_digest not in COMPATIBLE_IO_CODE_SHA256S:raise RuntimeError('Frozen worker scientific code changed')
                backup=frozen.with_name('runtime_single.before_'+frozen_digest[:12]+'.py')
                if not backup.exists():shutil.copy2(frozen,backup)
                temporary=frozen.with_name('runtime_single.updating.tmp')
                shutil.copy2(Path(__file__).resolve(),temporary);os.replace(temporary,frozen)
                write(self.root/'io_repair_migration.json',dict(previous_code=frozen_digest,current_code=SCIENTIFIC_CODE_SHA256,
                      base_molecular_protocol_changed=False,fixed_production_segments_per_replica=1,
                      initialization_repair='exact constrained minimization, measured every-step position/velocity pilot gates, bounded warmup retries and original-timestep confirmation',
                      repair='fixed-cost sampling: statistical warnings reported without MD extensions; GP/integration unchanged',time=time.asctime()))
        write(self.root/'campaign_config.json',self.cfg)
        write(self.root/'adapter_settings.json',self.settings)
        self.state_file=self.root/'md_cache.json'
        self.state=read(self.state_file) if self.state_file.exists() else {'node_windows':{}}
        # A clean fixed-cost campaign must not silently borrow older windows
        # that used variable-length MD. Its own md_cache.json still resumes.
        self.reusable={}
        # Re-run the independent estimator and actual-molecule derivative tests.
        tests=dict(mathematical=selftest(),molecular=molecular_derivative_test(oldbundle))
        write(self.root/'estimator_tests.json',tests)
        versions=self.root/'source_versions';versions.mkdir(exist_ok=True)
        script_sha=digest(Path(__file__));snapshot=versions/f'{script_sha}.py'
        if not snapshot.exists():shutil.copy2(Path(__file__).resolve(),snapshot)
        write(self.root/'source_audit.json',dict(SOURCE_AUDIT,script_sha256=script_sha,
              scientific_code_sha256=SCIENTIFIC_CODE_SHA256,max_parallel_windows=MAX_PARALLEL_WINDOWS))
        for cached_summary in sorted((self.root/'windows').glob('window_*/force_summary.json')):
            compact_accepted_window(cached_summary.parent)
        self.status('ready',reusable_completed_windows=len(self.reusable),controller_settings=_ala_original_parameters())
        log('[SINGLE FILE] Attached original controller retained; fixed phi/psi, no springs; local CP2K I/O.')
        log('[PRESERVED SETTINGS] '+json.dumps(_ala_original_parameters()))

    def status(self,stage,**extra):
        _single_boot_status(stage,**extra)
        details=' '.join(f'{k}={extra[k]}' for k in ('paid_nodes','array_job','completed_windows','total_windows','error') if k in extra)
        log(f'[CONTROLLER] {stage} {details}')

    @staticmethod
    def key(center):return tuple(np.round(wrapped(np.asarray(center)[:2]),10))

    def bind(self,P,anchor,n12,n13,n23):
        self.P=np.asarray(P,float);self.anchor=int(anchor)
        self.n12,self.n13,self.n23=n12,n13,n23
        grid_sha=hashlib.sha256(self.P.tobytes()).hexdigest()
        if self.state.get('grid_sha256',grid_sha)!=grid_sha:raise ValueError('Grid changed')
        self.state.update(grid_sha256=grid_sha,anchor_node=int(anchor))
        write(self.state_file,self.state)
        self.tree=_ala_tree(self.P)

    def reference(self,points):
        if self.reference_interpolator is None:
            data=np.loadtxt(self.root/'simulations_essentials'/self.settings['reference_file'],comments='#')
            if data.ndim!=2 or data.shape[1]<3 or not np.isfinite(data[:,:3]).all():
                raise ValueError('Invalid/nonfinite external diagnostic FES')
            p=wrapped(data[:,:2]);p[np.isclose(np.abs(p),np.pi,atol=1e-7,rtol=0)]=-np.pi
            p=np.round(p,8);phi=np.unique(p[:,0]);psi=np.unique(p[:,1])
            ix=np.searchsorted(phi,p[:,0]);iy=np.searchsorted(psi,p[:,1])
            sums=np.zeros((len(psi),len(phi)));counts=np.zeros_like(sums)
            np.add.at(sums,(iy,ix),data[:,2]);np.add.at(counts,(iy,ix),1)
            if np.any(counts==0):raise ValueError('External diagnostic FES must form a complete grid')
            scale={'kcal/mol':4.184/self.kbt,'kJ/mol':1/self.kbt,'kBT':1}[self.settings['reference_units']]
            values=sums/counts*scale
            self.reference_interpolator=_AlaInterpolator(
                (np.r_[psi-2*np.pi,psi,psi+2*np.pi],np.r_[phi-2*np.pi,phi,phi+2*np.pi]),
                np.tile(values,(3,3)),bounds_error=True)
        p=np.asarray(points,float);shape=p.shape[:-1];q=wrapped(p.reshape(-1,2))
        return self.reference_interpolator(q[:,::-1]).reshape(shape)

    def validation(self,count,xmu,xsd,reference_anchor):
        # Diagnostics are off-grid in both CVs. No query/path nodes are withheld;
        # the external FES never supplies labels, scores or acceptance criteria.
        points=wrapped(self.P[:,:2]+np.pi/self.n)
        rng=np.random.default_rng(SEED+9191)
        points=points[rng.choice(len(points),min(count,len(points)),replace=False)]
        if np.min(self.tree.query(points+np.pi)[0])<1e-8:raise AssertionError('Diagnostic point intersects paid grid')
        target=self.reference(points)-reference_anchor
        X=standardize_apply(_ala_features(points),xmu,xsd)
        np.savez(self.root/'external_reference_diagnostic_points.npz',phi_psi_rad=points,reference_kbt=target)
        return X,target

    def _read(self,g,wid):
        mean,draws=estimate(self.root,wid,self.cfg,self.settings)
        self.gradients[g]=mean/self.kbt;self.draws[g]=draws/self.kbt
        summary=read(self.root/'windows'/f'window_{wid:06d}'/'force_summary.json')
        self.covariances[g]=np.array(summary['covariance_kj2_mol2_rad2'])/self.kbt**2

    def query(self,nodes):
        requested=sorted(set(map(int,nodes)));mapping=self.state['node_windows']
        if len(mapping)+sum(str(g) not in mapping for g in requested)>self.settings['max_paid_nodes']:
            raise _AlaBudgetStop('Next connected batch exceeds configured MD-node budget; data retained')
        for g in requested:
            if g in self.gradients:continue
            if str(g) in mapping:
                wid=int(mapping[str(g)]);wd=self.root/'windows'/f'window_{wid:06d}'
                if (wd/'force_summary.json').exists() and read(wd/'force_summary.json')['quality_pass']:
                    self._read(g,wid)
                continue
            cached=self.reusable.get(self.key(self.P[g]))
            if cached is not None:
                wid=len(mapping);wd=self.root/'windows'/f'window_{wid:06d}';wd.mkdir(parents=True,exist_ok=True)
                summary=read(cached/'force_summary.json')
                if np.max(np.abs(wrapped(np.array(summary['center_rad'])-self.P[g,:2])))>1e-8:
                    raise ValueError('Reuse coordinate mismatch')
                for rep in range(self.cfg['replicas_per_window']):
                    link=wd/f'rep_{rep:02d}'
                    if not link.exists():link.symlink_to(cached/f'rep_{rep:02d}',target_is_directory=True)
                summary.update(window=wid,reused_from=str(cached))
                write(wd/'force_summary.json',summary)
                write(wd/'window.json',dict(id=wid,node=g,center=self.P[g,:2].tolist(),parent_window=None,reused_from=str(cached)))
                mapping[str(g)]=wid;write(self.state_file,self.state)
                self._read(g,wid);log(f'[MD REUSE] node={g} from {cached.name}')
        todo=set(requested)-set(self.gradients)
        # Plan all path windows against the already completed network, so no
        # pending window depends on another pending window finishing first.
        connected=set(self.gradients)
        if self.anchor in todo:connected.add(self.anchor)
        frontier=list(connected);allowed=connected|todo
        for g in frontier:
            for h in _ala_neighbors(g,self.n12,self.n13,self.n23):
                if h in allowed and h not in connected:connected.add(h);frontier.append(h)
        if not todo.issubset(connected):raise RuntimeError('MD path batch is disconnected from the network')
        ready=set(self.gradients);wids=[]
        for g in sorted(todo):
            parent=min(ready,key=lambda h:(np.linalg.norm(wrapped(self.P[g,:2]-self.P[h,:2])),h)) if ready else None
            if str(g) not in mapping:
                mapping[str(g)]=len(mapping);write(self.state_file,self.state)
            wid=int(mapping[str(g)]);wd=self.root/'windows'/f'window_{wid:06d}';wd.mkdir(parents=True,exist_ok=True)
            meta=dict(id=wid,node=g,center=self.P[g,:2].tolist(),
                      parent_window=None if parent is None else int(mapping[str(parent)]))
            if (wd/'window.json').exists():
                old=read(wd/'window.json')
                if old['node']!=g or not np.allclose(old['center'],meta['center'],atol=1e-12):
                    raise ValueError('Cached metadata mismatch')
            else:write(wd/'window.json',meta)
            source_trajectory=None;saved=read(wd/'window.json')
            if saved['parent_window'] is not None:
                rd=self.root/'windows'/f"window_{int(saved['parent_window']):06d}"/'rep_00'
                source_trajectory=sorted(rd.glob('segment_*/RUN_COMPLETE'))[-1].parent/'trajectory.xyz'
            _ala_prepare_window_start(self.root,wd,source_trajectory,meta['center'],self.model)
            wids.append(wid)
            log(f'[MD PLAN] W{wid:06d} node={g} phi/psi={meta["center"]}; independent completed parent={saved["parent_window"]}')
        _ala_parallel_dispatch(self,wids)
        for g in sorted(todo):self._read(g,int(mapping[str(g)]))
        gate=self.root/'seed_launch_gate.json'
        if not gate.exists() or read(gate).get('status')!='PASS':
            seeds,_=_ala_seed_geometry(self.anchor,self.n12,self.n13,self.n23)
            if set(map(int,seeds)).issubset(self.gradients):self.seed_gate(seeds)
        self.check_periodic_lines()
        return np.array([self.gradients[int(g)] for g in nodes])

    def seed_gate(self,seeds):
        # Include the seam in a complete phi turn. Periodic free energy should
        # have zero loop work within sampling and finite-grid quadrature error.
        seeds=list(map(int,seeds));value=variance=0.
        for i,g in enumerate(seeds):
            prev,nxt=seeds[(i-1)%len(seeds)],seeds[(i+1)%len(seeds)]
            c=.5*(wrapped(self.P[nxt,:2]-self.P[g,:2])+wrapped(self.P[g,:2]-self.P[prev,:2]))
            value+=float(c@self.gradients[g]);variance+=float(c@self.covariances[g]@c)
        se=float(np.sqrt(max(0.,variance)))
        report=dict(nodes=seeds,closed_psi_line_work_kbt=value,SE_kbt=se,seam_included=True,
                    status='PASS' if abs(value)<=max(.5,4*se) else 'FAIL')
        write(self.root/'seed_launch_gate.json',report)
        if report['status']!='PASS':log('[PERIODIC CLOSURE WARNING] Initial psi line: fixed-cost MD data retained; no extension')
        self.status('entire initial right-edge psi line accepted; original GAPS controller continues',paid_nodes=len(seeds))

    def check_periodic_lines(self):
        path=self.root/'periodic_line_checks.json'
        checks=read(path) if path.exists() else {}
        for axis in (0,1):
            for fixed in range(self.n):
                key=('phi' if axis==0 else 'psi')+f'_{fixed:02d}'
                if checks.get(key,{}).get('status')=='PASS':continue
                nodes=[ijk_to_g(0,k,fixed,1,self.n,self.n) if axis==0 else
                       ijk_to_g(0,fixed,k,1,self.n,self.n) for k in range(self.n)]
                if not set(nodes).issubset(self.gradients):continue
                value=variance=0.
                for i,g in enumerate(nodes):
                    prev,nxt=nodes[(i-1)%len(nodes)],nodes[(i+1)%len(nodes)]
                    c=.5*(wrapped(self.P[nxt,:2]-self.P[g,:2])+wrapped(self.P[g,:2]-self.P[prev,:2]))
                    value+=float(c@self.gradients[g]);variance+=float(c@self.covariances[g]@c)
                se=float(np.sqrt(max(0.,variance)))
                checks[key]=dict(nodes=nodes,closed_work_kbt=value,SE_kbt=se,seam_included=True,
                                 status='PASS' if abs(value)<=max(.5,4*se) else 'FAIL')
                write(path,checks)
                if checks[key]['status']!='PASS':
                    log(f'[PERIODIC CLOSURE WARNING] {key}: fixed-cost MD data retained; no extension')

    def edge_work(self,p0,p1):
        _,a=self.tree.query(wrapped(np.asarray(p0)[:2])+np.pi)
        _,b=self.tree.query(wrapped(np.asarray(p1)[:2])+np.pi)
        a,b=int(a),int(b)
        if a not in self.gradients or b not in self.gradients:raise RuntimeError('Unpaid MD edge endpoint')
        return float(.5*np.dot(self.gradients[a]+self.gradients[b],wrapped(np.asarray(p1)[:2]-np.asarray(p0)[:2])))

    def record_labels(self,nodes,anchor,P,U,adj):
        loc={int(g):i for i,g in enumerate(nodes)};start=loc[int(anchor)]
        dist=np.full(len(nodes),np.inf);dist[start]=0.;check=np.full(len(nodes),np.nan);check[start]=0.
        draws=np.zeros((self.settings['bootstrap_draws'],len(nodes)))
        parents=np.full(len(nodes),-1,int);heap=[(0.,start)]
        while heap:
            d,u=heapq.heappop(heap)
            if d!=dist[u]:continue
            a=int(nodes[u])
            for v,length,work in adj[u]:
                nd=d+length
                if nd<dist[v]:
                    b=int(nodes[v]);dist[v]=nd;check[v]=check[u]+work;parents[v]=a
                    delta=wrapped(P[b,:2]-P[a,:2])
                    draws[:,v]=draws[:,u]+.5*np.sum((self.draws[a]+self.draws[b])*delta,axis=1)
                    heapq.heappush(heap,(nd,v))
        np.testing.assert_allclose(check,U,atol=1e-10,rtol=1e-10)
        residual=[]
        for u,edges in enumerate(adj):
            for v,length,work in edges:
                if u<v:residual.append(float(U[v]-U[u]-work))
        folder=self.root/'integrated_labels';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'paid_{len(nodes):05d}.npz',nodes=nodes,energy_kbt=U,
                            energy_draws_kbt=draws,energy_SE_kbt=draws.std(axis=0,ddof=1),
                            parent_nodes=parents,closure_residuals_kbt=residual)
        self.status('original GAPS fit/acquisition',paid_nodes=len(nodes),
                    edge_closure_RMS_kbt=float(np.sqrt(np.mean(np.square(residual)))) if residual else 0.)


def _ala_draw_periodic_edge(ax,a,b,**kwargs):
    for first,last in _single_periodic_plot_segments(a,b):
        ax.plot([first[0],last[0]],[first[1],last[1]],**kwargs)


def _ala_plot_step(step_name,model,plot_g,P_full,P_feas,feas_pos,Xs_feas,xmu,xsd,ymu,ysd,normU,
                   U_oracle_flat,U_oracle_diag,U_oracle_face,train_all_set,train_q_set,val_g,anchor_g,
                   *legacy_args,**kwargs):
    mean,std=predict_mu_std_z(model,Xs_feas);mean=z2y(mean,ymu,ysd);std=std*ysd
    n=_ALA.n;axis=np.linspace(-np.pi,np.pi,n,endpoint=False)
    train=np.array(sorted(train_all_set),int)
    edges=build_edges_from_nodes(train_all_set,1,n,n,np.ones(len(P_full),bool))
    mst=mst_edges_over_allowed_edges(P_full,train,edges)
    reference=np.asarray(U_oracle_flat)
    # P_feas order is phi-major, psi-minor; transpose for phi on horizontal axis.
    values=[mean.reshape(n,n).T,std.reshape(n,n).T,np.abs(mean-reference).reshape(n,n).T]
    labels=[r'$F^{GP}\;(k_{\mathrm{B}}T)$',r'$\sigma_{GP}\;(k_{\mathrm{B}}T)$',
            r'$|F^{GP}-F^{reference}|\;(k_{\mathrm{B}}T)$']
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    for ax,z,label in zip(axes,values,labels):
        pc=ax.pcolormesh(axis,axis,z,shading='auto',cmap='viridis')
        fig.colorbar(pc,ax=ax,label=label);ax.set_xlabel(r'$\phi$ (rad)');ax.set_ylabel(r'$\psi$ (rad)')
        ax.set_xlim(-np.pi,np.pi);ax.set_ylim(-np.pi,np.pi);ax.set_aspect('equal')
    for a,b in mst:_ala_draw_periodic_edge(axes[0],P_full[a,:2],P_full[b,:2],color='lightgrey',lw=.8,alpha=.8)
    displayed=_single_plot_coordinates(P_full[train,:2])
    axes[0].scatter(displayed[:,0],displayed[:,1],s=12,c='tab:orange',edgecolors='k',linewidths=.2,clip_on=False)
    _single_plot_anchor(axes[0],P_full[int(anchor_g),:2])
    previous_show=plt.show
    try:
        plt.show=lambda *a,**k:None
        savefig(fig,f'Alanine_{step_name}_FES_sigma_external_error')
    finally:
        plt.show=previous_show
        plt.close(fig)
    folder=_ALA.root/'analysis';folder.mkdir(exist_ok=True)
    scores=kwargs.get('score_map_for_this_step') or {}
    np.savez_compressed(folder/f'step_{step_name}.npz',phi_psi_rad=P_full[:,:2],mean_kbt=mean,sigma_kbt=std,
                        external_reference_kbt=reference,paid_nodes=train,queried_nodes=sorted(train_q_set),
                        mst_edges=np.asarray(mst,int),candidate_nodes=list(scores),candidate_scores=list(scores.values()),
                        anchor_node=int(anchor_g))
    write(folder/f'step_{step_name}.json',dict(step=step_name,paid_nodes=len(train),anchor_node=int(anchor_g),
        external_reference_RMSE_kbt=float(np.sqrt(np.mean((mean-reference)**2))),
        metric='External-FES diagnostic; not held-out MD validation',reference_units_assumed=_ALA.settings['reference_units']))
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            _single_display_analysis(folder/f'step_{step_name}.npz',_SPYDER_PLOT_HISTORY,_SPYDER_PLOT_VIEWER)
        else:log(f'[PLOT] Saved step {step_name} figures and analysis')
    except Exception as exc:
        log('[PLOT WARNING] '+str(exc))


def _ala_enable(args):
    global _ALA,N3,D12_MIN,D12_MAX,D13_MIN,D13_MAX,D23_MIN,D23_MAX,RUN_TAG
    global DO_FULLGRID_LSQR_DIAGNOSTIC,LAB_X,LAB_Y,LAB_Z,LAB_W3_TRUE,LAB_W3_PRED,LAB_C
    global XLIM,YLIM,ZLIM,plot_scatter3d_color,plot_slice_2d_heat,plot_slice_3d_surface,plot_face_2d_heat,plot_face_3d_surface
    args.label_source='pathint';args.target_mode='umb';args.symmetry_mode='none';args.kernel_mode='plain'
    _ALA=_SingleAlanineAdapter();N3=_ALA.n
    D12_MIN=D12_MAX=0.;D13_MIN=D23_MIN=-np.pi;D13_MAX=D23_MAX=np.pi
    XLIM=YLIM=(-np.pi,np.pi);ZLIM=(-.5,.5)
    RUN_TAG='Alanine_RIGHT_EDGE_B5_COMPACT_v18_v17';DO_FULLGRID_LSQR_DIAGNOSTIC=False
    LAB_X=r'$\phi\;(rad)$';LAB_Y=r'$\psi\;(rad)$';LAB_Z='unused storage coordinate'
    LAB_W3_TRUE=r'$F^{reference}\;(k_{\mathrm{B}}T)$';LAB_W3_PRED=LAB_C=r'$F^{GP}\;(k_{\mathrm{B}}T)$'
    # Original PIP-only scatter/decomposition/slice startup pages have no
    # physical interpretation for a two-angle system. Alanine plots are added
    # through a guarded branch in the original per-step plotting function.
    def skip_pip_only_page(*a,**k):return None
    plot_scatter3d_color=plot_slice_2d_heat=plot_slice_3d_surface=skip_pip_only_page
    plot_face_2d_heat=plot_face_3d_surface=skip_pip_only_page
    mpl.rcParams['text.usetex']=False


def _single_entry():
    if '--original' in sys.argv:
        sys.argv.remove('--original');main();return
    sys.argv=[str(Path(__file__).resolve())]
    try:
        main()
    except KeyboardInterrupt:
        if _ALA is not None:_ALA.status('PAUSED; submitted MD workers continue')
        else:_single_boot_status('PAUSED during startup')
        log('Controller paused. Completed samples are retained; rerun this file to resume.')
    except _AlaBudgetStop as exc:
        if _ALA is not None:_ALA.status('configured paid-node budget reached',message=str(exc))
        log('[MD BUDGET STOP] '+str(exc))
    except BaseException as exc:
        if _ALA is not None:_ALA.status('STOPPED',error=str(exc))
        elif _SPYDER_RUN_ID is not None:_single_boot_status('STOPPED during controller initialization',error=str(exc))
        raise
    else:
        if _ALA is not None:_ALA.status('original GAPS controller finished')
    finally:
        if _ALA is not None:
            try:_ALA.lock.close()
            except Exception:pass
        _single_close_controller()



if __name__=='__main__':
    _single_entry()
