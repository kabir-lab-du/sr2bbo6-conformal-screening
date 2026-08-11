#!/bin/bash
# Sr2GaSbO6 HSE06 static on the experimental I4/m structure. 48 MPI ranks.
# --bind-to none: the box is oversubscribed (shared with colleagues' VASP jobs);
# floating ranks lets the kernel balance instead of piling onto cores 0-47.
set -u
cd "$(dirname "$0")"
# Runtime env copied from a live production vasp_std process (non-interactive ssh
# shells lack it; VASP 6.4.3 is linked against this HDF5/MKL/OpenMPI stack).
export LD_LIBRARY_PATH=/home/alamgir/VASP/vasp.6.5.1/src/oneapi/lib:/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:/home/alamgir/VASP/new/hdf5-1.14.4-2/build/lib:/home/alamgir/VASP/new/openmpi-5.0.0/build/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
# Polite MPI polling: on this oversubscribed box (load ~150 on 96 cores) hard
# busy-wait wedged the first attempt — one rank starved in recvfrom for 9 h
# while 47 spun in a collective. Yielding degrades peak speed slightly but
# prevents the livelock.
export OMPI_MCA_mpi_yield_when_idle=1
VASP=/home/alamgir/VASP/new/vasp.6.4.3/bin/vasp_std
MPI=/home/alamgir/VASP/new/openmpi-5.0.0/build/bin/mpirun
NP=16
echo "[run] $(date '+%F %T') Sr2GaSbO6 HSE06 static starting, np=$NP"
$MPI -np $NP --bind-to none --mca pml ob1 --mca btl self,sm --mca coll ^han,adapt "$VASP" > vasp.log 2>&1
if [ -f OUTCAR ] && grep -q "fundamental gap" OUTCAR; then
  grep -m1 "fundamental gap" OUTCAR | tee RESULT.txt
  echo "[run] $(date '+%F %T') COMPLETE" | tee -a RESULT.txt
else
  echo "[run] $(date '+%F %T') ERROR: no fundamental gap in OUTCAR — see vasp.log"
  exit 1
fi
