import re
import os
import glob2
import numpy as np
import os.path as pa
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, Column
from sfft.utils.ReadWCS import Read_WCS
from sfft.utils.meta.MultiProc import Multi_Proc
from multiprocessing import Pool
from tqdm import tqdm
# Epy310

MODE = "SingleSCI"
GDIR = "/hildafs/projects/phy220048p/share/decam_pipeline/DECAM_FOLLOWUP"

if MODE == "StackDIFF":
    FITS_DIFFPHOT_Queue = glob2.glob(
        GDIR + "/DATA/S251112cm/HP*/DECam-*/202*-*-*/sc4d_*/cache/sc4d_*.skysub.resampled.diff.phot/sc4d_*.[N|S]*.skysub.resampled.diff.phot.fits"
    )

if MODE == "SingleSCI":
    FITS_DIFFPHOT_Queue = glob2.glob(
        GDIR + "/DATA/S251112cm/HP*/DECam-*/202*-*-*/c4d_*/cache/c4d_*.skysub.resampled.phot/c4d_*.[N|S]*.skysub.resampled.phot.fits"
    )

def func_retrieve_row(idx):
    FITS_DIFFPHOT = FITS_DIFFPHOT_Queue[idx]
    EXPNAME = re.split('\.', pa.basename(FITS_DIFFPHOT))[0]

    if MODE == "StackDIFF":
        try:  
            dat, mom = re.split('_', pa.basename(FITS_DIFFPHOT))[1:3]
            dat = dat[-6:]    # a patch
            time = "20%s-%s-%sT%s:%s:%s" % (dat[0:2], dat[2:4], dat[4:6], mom[0:2], mom[2:4], mom[4:6])
            MJD_OBS0 = float(Time(time, format='isot').mjd)   # MJD of stack 

            MJD_OBS = []
            for FITS in glob2.glob(pa.dirname(pa.dirname(pa.dirname(pa.dirname(FITS_DIFFPHOT)))) + "/c4d_*/cache/c4d_*.skysub.resampled/c4d_*.primary.skysub.resampled.fits"):
                MJD_OBS.append(fits.getheader(FITS, ext=0)["MJD-OBS"])
            MJD_OBS = np.mean(MJD_OBS)
            assert np.abs(MJD_OBS - MJD_OBS0) < 0.01, f"Time difference between {FITS_DIFFPHOT} and the stack is too large."

            FILTER = re.split('_', EXPNAME)[-2]
            assert FILTER == pa.dirname(pa.dirname(pa.dirname(pa.dirname(pa.dirname(FITS_DIFFPHOT)))))[-1]

            AstDP = Table.read(FITS_DIFFPHOT)
            DETPOS = re.split('\.', pa.basename(FITS_DIFFPHOT))[1]
            assert DETPOS == AstDP.meta["DETPOS"]
            LM5_DIFF = AstDP.meta["LM5"]

            FITS_DIFF = re.sub(".skysub.resampled.diff.phot", ".skysub.resampled.diff", FITS_DIFFPHOT)
            assert pa.exists(FITS_DIFF), f"File {FITS_DIFF} does not exist. Please check the path."

            phdr_DIFF = fits.getheader(FITS_DIFF, ext=0)
            w = Read_WCS.RW(hdr=phdr_DIFF)
            RA_CCD, DEC_CCD = w.all_pix2world(np.array([[phdr_DIFF['NAXIS1'] / 2, phdr_DIFF['NAXIS2'] / 2]]) + 0.5, 1)[0]
            row = [EXPNAME, DETPOS, MJD_OBS, FILTER, RA_CCD, DEC_CCD, LM5_DIFF]
        except:
            row = None
    
    if MODE == "SingleSCI":
        try:
            LM5_SCI = Table.read(FITS_DIFFPHOT).meta["LM5"]
            FITS_SCI = re.sub(".skysub.resampled.phot", ".skysub.resampled", FITS_DIFFPHOT)
            phdr_SCI = fits.getheader(FITS_SCI, ext=0)
            MJD_OBS = phdr_SCI["MJD-OBS"]
            FILTER = phdr_SCI["FILTER"][0]
            DETPOS = phdr_SCI["DETPOS"]
            w = Read_WCS.RW(hdr=phdr_SCI)
            RA_CCD, DEC_CCD = w.all_pix2world(np.array([[phdr_SCI['NAXIS1'] / 2, phdr_SCI['NAXIS2'] / 2]]) + 0.5, 1)[0]
            row = [EXPNAME, DETPOS, MJD_OBS, FILTER, RA_CCD, DEC_CCD, LM5_SCI]
        except:
            row = None
    return row

Ntask = len(FITS_DIFFPHOT_Queue)
taskid_lst = np.arange(Ntask)
print(f"Total {Ntask} tasks to process.")
nproc = 64

with Pool(processes=nproc) as pool:
    results_iter = pool.imap_unordered(func_retrieve_row, taskid_lst, chunksize=1)
    results = [out for out in tqdm(results_iter, total=len(taskid_lst), desc="Retrieving rows")]

rowdata = [row for row in results if row is not None]
AstDEP = Table(rows=rowdata, names=['EXPNAME', 'DETPOS', 'MJD_OBS', 'FILTER', 'RA_CCD', 'DEC_CCD', 'LM5'])
AstDEP.add_column(Column(Time(AstDEP['MJD_OBS'], format='mjd').isot), name='DATE_OBS', index=2)
AstDEP.write("S251112cm_DECam_Depth_%s.csv" % MODE, format='ascii.csv', overwrite=True)