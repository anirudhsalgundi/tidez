import os
import glob
import os.path as pa
import pwd

GDIR = '/ocean/projects/phy250024p/shared/DECAM_ZTF/DECAM_FOLLOWUP'
PROGRAM = 'DECAM_ZTF_ALL_2025B'
DRY_RUN = False   # set to False to actually delete

print('GDIR:', GDIR)
print('PROGRAM:', PROGRAM)
print('DRY_RUN:', DRY_RUN)

# remove unskysubtracted:unresampled science
CDIR_LIST1 = [
    cdir for cdir in glob.glob(GDIR + '/DATA/%s/*/DECam-*/20*-*-*/c4d_*/cache/c4d_*' % PROGRAM)
    if 'skysub' not in pa.basename(cdir)
]

# remove skysubtracted:unresampled science
CDIR_LIST2 = glob.glob(GDIR + '/DATA/%s/*/DECam-*/20*-*-*/c4d_*/cache/c4d_*.skysub' % PROGRAM)

# remove unskysubtracted template
CDIR_LIST3 = [
    cdir for cdir in glob.glob(GDIR + '/DATA/%s/*/DECam-*/TEMPLATE/c4d_*/cache/c4d_*' % PROGRAM)
    if 'skysub' not in pa.basename(cdir)
]

ALL_CDIRS = CDIR_LIST1 + CDIR_LIST2 + CDIR_LIST3
print('Cache dirs found:', len(ALL_CDIRS))

deleted_bytes = 0
deleted_files = 0
failed_files = 0
skipped = {}   # owner -> [n_dirs, n_bytes]

for cdir in ALL_CDIRS:
    files = [f for f in glob.glob(cdir + '/*') if pa.isfile(f)]
    if len(files) == 0:
        continue

    dir_bytes = sum(pa.getsize(f) for f in files)

    # deletion needs write permission on the directory, not the file
    if not os.access(cdir, os.W_OK):
        uid = os.stat(cdir).st_uid
        try:
            owner = pwd.getpwuid(uid).pw_name
        except KeyError:
            owner = 'uid_%d' % uid
        if owner not in skipped:
            skipped[owner] = [0, 0]
        skipped[owner][0] += 1
        skipped[owner][1] += dir_bytes
        continue

    for f in files:
        size = pa.getsize(f)
        if DRY_RUN:
            deleted_bytes += size
            deleted_files += 1
            continue
        try:
            os.remove(f)
            deleted_bytes += size
            deleted_files += 1
        except OSError as e:
            failed_files += 1
            print('FAILED:', f, e)

label = 'Would delete' if DRY_RUN else 'Deleted'
print('\n%s: %d files, %.2f GB' % (label, deleted_files, deleted_bytes / 1e9))
print('Failed: %d files' % failed_files)

print('\nSkipped (no write permission on directory):')
for owner in sorted(skipped, key=lambda o: skipped[o][1], reverse=True):
    n_dirs, n_bytes = skipped[owner]
    print('  %-15s %6d dirs  %10.2f GB' % (owner, n_dirs, n_bytes / 1e9))