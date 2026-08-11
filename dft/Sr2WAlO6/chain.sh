#!/bin/bash
# Sr2WAlO6 pipeline: PBE+U relax (FM and AFM) -> pick lower-E0 magnetic order
# -> HSE06 static gap on the winner's relaxed structure. 48 MPI ranks per stage.
# --bind-to none: box is shared and oversubscribed; let the kernel balance ranks.
# Re-runnable: finished relax stages are skipped via DONE flags.
set -u
cd "$(dirname "$0")"
# Runtime env copied from a live production vasp_std process (non-interactive ssh
# shells lack it; VASP 6.4.3 is linked against this HDF5/MKL/OpenMPI stack).
export LD_LIBRARY_PATH=/home/alamgir/VASP/vasp.6.5.1/src/oneapi/lib:/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:/home/alamgir/VASP/new/hdf5-1.14.4-2/build/lib:/home/alamgir/VASP/new/openmpi-5.0.0/build/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
# Polite MPI polling — both hard-busy-wait attempts on this box livelocked
# (GaSbO6: one rank starved in recvfrom 9 h; MnWO6 AFM: all-rank collective
# deadlock, files frozen 9 h). Yield prevents the livelock at slight peak cost.
export OMPI_MCA_mpi_yield_when_idle=1
VASP=/home/alamgir/VASP/new/vasp.6.4.3/bin/vasp_std
MPI=/home/alamgir/VASP/new/openmpi-5.0.0/build/bin/mpirun
NP=16

for m in FM AFM; do
  d=relax_$m
  mkdir -p "$d"
  [ -f "$d/DONE" ] && { echo "[chain] $d already done, skipping"; continue; }
  cp POSCAR_init  "$d/POSCAR"
  cp "INCAR_relax_$m" "$d/INCAR"
  cp KPOINTS      "$d/KPOINTS"
  cp POTCAR       "$d/POTCAR"
  echo "[chain] $(date '+%F %T') relax_$m starting (np=$NP)"
  ( cd "$d" && $MPI -np $NP --bind-to none --mca pml ob1 --mca btl self,sm --mca coll ^han,adapt "$VASP" > vasp.log 2>&1 )
  if grep -q "reached required accuracy" "$d/vasp.log" 2>/dev/null || \
     grep -q "reached required accuracy" "$d/OUTCAR" 2>/dev/null; then
    touch "$d/DONE"
  elif grep -q "General timing and accounting" "$d/OUTCAR" 2>/dev/null; then
    touch "$d/DONE"
    echo "[chain] WARNING: relax_$m terminated without 'reached required accuracy' (NSW exhausted?)"
  else
    echo "[chain] ERROR: relax_$m failed — see $d/vasp.log"; exit 1
  fi
done

eFM=$(tail -1 relax_FM/OSZICAR  | sed 's/.*E0= *\([^ ]*\).*/\1/')
eAFM=$(tail -1 relax_AFM/OSZICAR | sed 's/.*E0= *\([^ ]*\).*/\1/')
W=$(awk -v a="$eFM" -v b="$eAFM" 'BEGIN{print (a+0 <= b+0) ? "FM" : "AFM"}')
echo "[chain] E0(FM)=$eFM  E0(AFM)=$eAFM  ->  winner: $W" | tee MAGNETIC_ORDER.txt

mkdir -p hse06
cp "relax_$W/CONTCAR" hse06/POSCAR
cp "INCAR_hse06_$W"   hse06/INCAR
cp KPOINTS            hse06/KPOINTS
cp POTCAR             hse06/POTCAR
echo "[chain] $(date '+%F %T') hse06 starting ($W order, np=$NP)"
( cd hse06 && $MPI -np $NP --bind-to none --mca pml ob1 --mca btl self,sm --mca coll ^han,adapt "$VASP" > vasp.log 2>&1 )
grep -m1 "fundamental gap" hse06/OUTCAR | tee RESULT.txt
echo "[chain] $(date '+%F %T') COMPLETE ($W)" | tee -a RESULT.txt
