#region: Modules.
from ase import Atoms
from ase.io import read, write 
from ase.build import make_supercell, molecule
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from ase.dft.kpoints import bandpath, get_special_points
import numpy as np 
import os 
import argparse
import time 
import subprocess
import glob 
import re 
#endregion

#region: Variables.

# TODO: Single wfn class and k-points as inputs. Additional steps in the workflow (phbands, xctph, esf). Test on Si. 
# TODO: Convergence. Test on Si. And on SiO2 too. 

total_time:float = 0.0
#endregion

#region: Functions.
def create_atoms(sc_grid):
    
    #region: Silicon. 
    # symbols = ['Si', 'Si']
    # A = 5.43
    # uc_atoms = Atoms(
    #     numbers=[atomic_numbers[symbol] for symbol in symbols],
    #     cell=np.array([
    #         [0.5, 0.5, 0.0],
    #         [0.0, 0.5, 0.5],
    #         [0.5, 0.0, 0.5],
    #     ])*A,
    #     scaled_positions=np.array([
    #         [0.00, 0.00, 0.00],
    #         [0.25, 0.25, 0.25],
    #     ]),
    #     pbc=[True, True, True],
    # )
    # sc_atoms = make_supercell(uc_atoms, np.diag(sc_grid))
    # sc_atoms.rattle()
    #endregion
    
    #region: H20
    uc_atoms = molecule('H2O', vacuum=3.0)
    sc_atoms = uc_atoms
    #endregion
    
    #region: Double-Perovskite. 
    # symbols = [
    #     'Cs',
    #     'Cs',
    #     'Ag',
    #     'In',
    #     'Cl',
    #     'Cl',
    #     'Cl',
    #     'Cl',
    #     'Cl',
    #     'Cl',
    # ]
    # A = 10.48
    # uc_atoms = Atoms(
    #     numbers=[atomic_numbers[symbol] for symbol in symbols],
    #     cell=np.array([
    #         [0.5, 0.5, 0.0],
    #         [0.0, 0.5, 0.5],
    #         [0.5, 0.0, 0.5],
    #     ])*A,
    #     scaled_positions=np.array([
    #         [0.25, 0.25, 0.25],
    #         [0.75, 0.75, 0.75],
    #         [0.50, 0.50, 0.50],
    #         [0.00, 0.00, 0.00],
    #         [0.25, 0.25, 0.75],
    #         [0.25, 0.75, 0.25],
    #         [0.75, 0.25, 0.25],
    #         [0.75, 0.75, 0.25],
    #         [0.75, 0.25, 0.75],
    #         [0.25, 0.75, 0.75],
    #     ]),
    #     pbc=[True, True, True],
    # )
    # # write('struct_ase_uc.cif', [atoms])        # Write the unit cell cif file.

    # # # Make supercell. 
    # sc_atoms = make_supercell(uc_atoms, np.diag(sc_grid))
    # sc_atoms.rattle()
    # write('struct_ase_sc.cif', [atoms])        # Write the supercell cell cif file.
    # atoms.rattle()
    # array = atoms.get_atomic_numbers()
    # array[2] = 11
    # atoms.set_atomic_numbers(array)
    # write('struct_ase_sc_rattled.cif', [atoms])        # Write the supercell cell cif file.
    #endregion
    
    #region: Save Structures. 
    write('uc_atoms.xsf', uc_atoms)
    write('sc_atoms.xsf', sc_atoms)
    #endregion 
    
    return uc_atoms, sc_atoms 

def create_kpath(uc_atoms, sc_grid, path_string):
    npoints = 100 
    
    # Create kpath, Kpath, Gshift variables.  
    bpath = bandpath(path_string, uc_atoms.cell, npoints)
    bpath.write('bandpath.json')
    kpath: np.ndarray = bpath.kpts    
    Kscaled = kpath
    Kscaled[:, 0] *= sc_grid[0]
    Kscaled[:, 1] *= sc_grid[1]
    Kscaled[:, 2] *= sc_grid[2]
    Kpath, Gshift = np.modf(Kscaled)
    
    # Write them to files. 
    np.savetxt('uc_kpath.txt', kpath)
    np.savetxt('sc_Kpath.txt', Kpath)
    np.savetxt('sc_Gshift.txt', Gshift)
    np.savetxt('sc_grid.txt', sc_grid)
    
    # Return sc grid. 
    return Kpath 
    
def create_input_dict():
    
    # scheduler = Summit()
    scheduler = Perlmutter()
    scheduler = WSL()
    single_task_desc = JobProcDesc(time='01:45', nodes=1, tasks=1)
    single_node_desc = JobProcDesc(time='01:45', nodes=1, tasks=64)
    para_desc = JobProcDesc(time='01:45', nodes=1, tasks=64)
    big_para_desc = JobProcDesc(time='01:45', nodes=1, tasks=64)
    
    sc_grid = np.array([1, 1, 1], dtype='i4')
    uc_atoms, sc_atoms = create_atoms(sc_grid)
    path_string = 'GG'
    
    # In case ESD calculations are done. Will override the sc_atoms with the one read from file. 
    if os.path.exists('./esd_atoms.xsf'):
        sc_atoms = read('./esd_atoms.xsf')
    
    atoms = AtomsInput(
        atoms = sc_atoms,
    )
    
    scf = ScfInput(
        kgrid=(1, 1, 1),
        ecutwfc=20.0,
        job_proc_desc=para_desc
    )
    
    dfpt = DfptInput(
        atoms=atoms,
        qgrid= (2, 2, 2),
        conv_threshold='1.0d-16',
        njobs=1,
        job_proc_desc=para_desc,
    )
    
    phbands = PhbandsInput(
        kpath=create_kpath(uc_atoms, sc_grid, path_string),
        job_proc_desc=single_node_desc,
    )
    
    phdos = PhDosInput(
        qdim=(2, 2, 2),
        job_proc_desc=single_node_desc,
    )
    
    phmodes = PhModesInput(
        qidx=1,     
        job_proc_desc=single_node_desc
    )
    
    dos = DosInput(
        bands=50,
        kdim=(2, 2, 2),
        job_proc_desc=para_desc
    )
    
    wannier = WannierInput(
        atoms=atoms,
        kdim=(2, 2, 2),
        kpath_str=None,
        num_bands=50,
        num_wann=50,
        job_proc_desc=single_task_desc
    )
    
    dftelbands = DftElbandsInput(
        kpath=create_kpath(uc_atoms, sc_grid, path_string),
        nbands=50,
        job_proc_desc=para_desc,
    )
    
    wfn = WfnGeneralInput(
        atoms=atoms,
        kdim=(2, 2, 2),
        qshift=(0.0, 0.0, 0.000),
        is_sym=False,
        bands=100,
        job_proc_desc=para_desc,
        job_proc_desc_pw2bgw=single_node_desc,
        job_proc_desc_parabands=para_desc,
        parabands_bands=201,
    )
    
    wfnq = WfnGeneralInput(
        atoms=atoms,
        kdim=(2, 2, 2),
        qshift=(0.0, 0.0, 0.001),
        is_sym=False,
        bands=50,
        job_proc_desc=para_desc,
        job_proc_desc_pw2bgw=single_node_desc,
    )
    
    wfnfi = WfnGeneralInput(
        atoms=atoms,
        kdim=(2, 2, 2),
        qshift=(0.0, 0.0, 0.0),
        is_sym=False,
        bands=50,
        job_proc_desc=para_desc,
        job_proc_desc_pw2bgw=single_node_desc,
    )
    
    epw = EpwInput(
        kgrid_coarse=(2, 2, 2),
        qgrid_coarse=(2, 2, 2),
        kgrid_fine=(2, 2, 2),
        qgrid_fine=(2, 2, 2),
        bands=50,
        exec_loc='$START_DIR/Fortran/QuantumEspresso/q-e-cpu/bin/epw.x',
        skipped_bands=(51, 100),
        job_proc_desc=single_task_desc,
    )
    
    wfnqfi = WfnGeneralInput(
        atoms=atoms,
        kdim=(2, 2, 2),
        qshift=(0.0, 0.0, 0.001),
        is_sym=False,
        bands=50,
        job_proc_desc=para_desc,
        job_proc_desc_pw2bgw=single_node_desc,
    )
    
    epsilon = EpsilonInput(
        wfn_input=wfn,
        bands=200,
        cutoff=10.0,
        wfn_link='WFN_parabands.h5',
        wfnq_link='WFNq.h5',
        job_proc_desc=para_desc,
    )
    
    sigma = SigmaInput(
        wfn_input=wfn,
        bands=200,
        band_min=1,
        band_max=14,
        cutoff=10.0,
        wfn_inner_link='WFN_parabands.h5',
        job_proc_desc=para_desc,
    )
    
    gwelbands = GwElbandsInput(
        val_bands_coarse=4,
        cond_bands_coarse=6,
        val_bands_fine=4,
        cond_bands_fine=5,
        wfn_co_link='WFN',
        wfn_fi_link='WFN_dft_elbands',
        job_proc_desc=single_node_desc,
    )
    
    kernel = KernelInput(
        val_bands_coarse=4,
        cond_bands_coarse=10,
        Qshift=(0.0, 0.0, 0.0),
        wfn_co_link='WFN_parabands.h5',
        job_proc_desc=para_desc,
    )
    
    absorption = AbsInput(
        val_bands_coarse=4,
        cond_bands_coarse=10,
        val_bands_fine=4,
        cond_bands_fine=10,
        Qshift=(0.0, 0.0, 0.0),
        wfn_co_link='WFN_parabands.h5',
        wfn_fi_link='WFN_parabands.h5',
        wfnq_fi_link='WFNq.h5',
        job_proc_desc=para_desc,
    )
    
    plotxct = PlotxctInput(
        hole_position=[0.5, 0.5, 0.5],
        supercell_size=[2, 2, 2],
        state=1,
        wfn_fi_link='WFN_parabands.h5',
        wfnq_fi_link='WFNq.h5',
        job_proc_desc=para_desc,
    )
    
    esf = EsfInput(
        job_proc_desc=single_task_desc,
    )
    
    savestep = SaveStepInput(
        job_proc_desc=single_task_desc,
    )
    
    esd = EsdInput(
        job_proc_desc=single_node_desc
    )
    
    input = Input(
        scheduler=scheduler,
        atoms=atoms,
        scf=scf,
        dfpt=dfpt,
        phbands=phbands,
        phdos=phdos,
        phmodes=phmodes,
        dos=dos,
        wannier=wannier,
        dftelbands=dftelbands,
        wfn=wfn,
        wfnq=wfnq,
        wfnfi=wfnfi,
        epw=epw,
        wfnqfi=wfnqfi,
        epsilon=epsilon,
        sigma=sigma,
        gwelbands=gwelbands,
        kernel=kernel,
        absorption=absorption,
        plotxct=plotxct,
        esf=esf,
        savestep=savestep,
        esd=esd,
    )
    # For some reason I need to include these lines below. Don't know why for now. 
    input.scheduler=scheduler
    input.atoms = atoms 
    
    return input 

def write_string_to_file(filename, string):
    with open(filename, 'w') as f:
        f.write(string)
        
def create_workflow(list_of_steps):

    for step in list_of_steps:
        step.create_files()

    write_string_to_file(
        'job_all.sh',
        f'''#!/bin/bash
python3 workflow.py --run &> workflow_run.out 
        '''
    )
    
    # Add exec permission to scripts. 
    for file in glob.glob('*.sh'):
        os.system(f'chmod u+x {file}') 

def run_and_wait_command(script, input):
    '''
    Run each script and write out some logging info. 
    '''
    global total_time

    start_time = time.time()
    print(f'Starting {script}.', flush=True)
    ps_result = subprocess.run(f'{input.scheduler.get_job_prefix()}{script}')
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    total_time += elapsed_time

    if ps_result.returncode == 0:  # Success.
        print(f'Done with {script} in {elapsed_time:15.10f} seconds.\n\n', flush=True)
    else:               # Fail.
        print(f'Error finishing: {script}. Exited with code {ps_result.returncode}. Time elapsed is {elapsed_time:15.10f} seconds.\n\n', flush=True)
        print(f'Total time for workflow run in {total_time:15.10f} seconds.\n', flush=True)
        os._exit(ps_result.returncode)
        
def run_workflow(list_of_steps):

    for step in list_of_steps:
        step.run()
        
    # Write the total workflow run time. 
    print(f'Done whole worflow in {total_time:15.10f} seconds.\n\n', flush=True)

def remove_workflow(list_of_steps):
    
    for step in list_of_steps:
        step.delete_files()
        
    os.system('rm -rf job_all.sh')
    os.system('rm -rf *.cif')

def save_step(folder_name, list_of_steps):
    
    for step in list_of_steps:
        step.save(folder_name)

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Script to manage workflows")

    # Add arguments
    parser.add_argument('--create', action='store_true', help="create workflow")
    parser.add_argument('--run', action='store_true', help="run workflow")
    parser.add_argument('--save', type=str, help="save workflow")
    parser.add_argument('--remove', action='store_true', help="remove workflow")

    # Parse the arguments
    args = parser.parse_args()
    
    # Perform actions based on arguments. 
    input = create_input_dict()
    
    list_of_steps = [
        Scf(input=input),
        # Relax(input=input),
        Md(input=input),
        # Dfpt(input=input),
        # PhBands(input=input),
        # PhDos(input=input),
        # PhModes(input=input),
        # Dos(input=input),
        # Pdos(input=input),
        # Kpdos(input=input),
        # Wannier(input=input),
        # DftElbands(input=input),
        # Wfn(input=input),
        # Epw(input=input),
        Wfnq(input=input),
        # Wfnfi(input=input),
        # Wfnqfi(input=input),
        # Epsilon(input=input),
        # Sigma(input=input),
        # GwElbands(input=input),
        # Kernel(input=input),
        # Absorption(input=input),
        # PlotXct(input=input),
        # Esf(input=input),
        # SaveStep(input=input),
        # Esd(input=input),
    ]
    
    if args.create:
        create_workflow(list_of_steps=list_of_steps)

    if args.run:
        run_workflow(list_of_steps=list_of_steps)

    if args.remove:
        remove_workflow(list_of_steps=list_of_steps) 
        
    if args.save:
        save_step(folder_name=args.save, list_of_steps=list_of_steps)
#endregion

#region: Classes.
class JobProcDesc:
    def __init__(self, time=None, nodes=None, tasks=None, nk=None, ni=None):
        self.time: str = time 
        self.nodes: int = nodes 
        self.tasks: int = tasks 
        self.nk: int = nk
        self.ni: int = ni 

class Scheduler:
    def __init__(self):
        pass 
    
    def get_job_header(self, job_proc_desc):
        return ''
    
    def get_parallel_prefix(self, job_proc_desc):
        return ''
    
    def get_parallel_infix(self, job_proc_desc: JobProcDesc):
        ni = '' if job_proc_desc.ni==None else f' -ni {job_proc_desc.ni}'
        nk = '' if job_proc_desc.nk==None else f' -ni {job_proc_desc.nk}'
        
        output = f'{ni} {nk}'
        
        return output 
    
    def get_job_prefix(self):
        return ''

class Frontera(Scheduler):
        def get_job_header(self, job_proc_desc: JobProcDesc):
            output = f'''#SBATCH --account=PHY20032
#SBATCH --queue=debug
#SBATCH --job-name=struct_job
#SBATCH --nodes={job_proc_desc.nodes}
#SBATCH --time={job_proc_desc.time}
#SBATCH --mail-type=all
$SBATCH --mail-user=krishnaa.vadivel@yale.edu
            '''
        
            return output 
    
        def get_parallel_prefix(self, job_proc_desc: JobProcDesc):
            output = f'ibrun -n{job_proc_desc.tasks} '
            
            return output 
        
        def get_job_prefix(self):
            return 'sbatch '
            # return 'sbatch --wait'      # If we want to wait for the job to complete. 

class Perlmutter(Scheduler):
        def get_job_header(self, job_proc_desc: JobProcDesc):
            output = f'''#SBATCH --account=m3571
#SBATCH --qos=debug
#SBATCH --job-name=struct_job
#SBATCH --nodes={job_proc_desc.nodes}
#SBATCH --time={job_proc_desc.time}
#SBATCH --mail-type=all
$SBATCH --mail-user=krishnaa.vadivel@yale.edu
            '''
        
            return output 
    
        def get_parallel_prefix(self, job_proc_desc: JobProcDesc):
            output = f'srun -n{job_proc_desc.tasks} '
            
            return output 
        
        def get_job_prefix(self):
            return 'sbatch '
            # return 'sbatch --wait'      # If we want to wait for the job to complete. 

class Summit(Scheduler):
    def get_job_header(self, job_proc_desc: JobProcDesc):
        output = f'''#BSUB -P cph156
#BSUB -q batch
#BSUB -J struct_job
#BSUB -nnodes {job_proc_desc.nodes}
#BSUB -W {job_proc_desc.time}
        '''
        
        return output 
    
    def get_parallel_prefix(self, job_proc_desc: JobProcDesc):
        output = f'jsrun -n{job_proc_desc.tasks} -a1 -c7 -g1 -bpacked:7 -EOMP_NUM_THREADS=28 '
        
        return output 
    
    def get_job_prefix(self):
        return 'bsub '
        # return 'bsub -K'      # If we want to wait for the job to complete. 

class WSL(Scheduler):
    pass 

class AtomsInput:
    def __init__(
        self,
        atoms,
    ):
        self.atoms:Atoms = atoms  
        
    def get_ntyp(self):
        return len(np.unique(self.atoms.get_atomic_numbers()))
    
    def get_nat(self):
        return self.atoms.get_number_of_atoms()

    def get_scf_cell(self):
        output = ''
        for row in self.atoms.get_cell():
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f}\n'
        return output 
    
    def get_scf_atomic_species(self):
        output = ''
        
        for atm_num in np.unique(self.atoms.get_atomic_numbers()):
            output += f'{chemical_symbols[atm_num]} {atomic_masses[atm_num]} {chemical_symbols[atm_num]}_ONCV_PBE_sr.upf\n'
        return output 

    def get_scf_atomic_positions(self, first_column='symbol'):
        output = ''
        
        if first_column=='symbol':
            for atm_num, row in zip(self.atoms.get_atomic_numbers(), self.atoms.get_positions()):
                output += f'{chemical_symbols[atm_num]} {row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f}\n'
        
        if first_column=='atom_index':
            _, atom_index = np.unique(self.atoms.get_atomic_numbers(), return_inverse=True)
            atom_index += 1     # 1 based index.
            for atm_num, row in zip(atom_index, self.atoms.get_positions()):
                output += f'{atm_num} {row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f}\n'
        return output 

class ScfInput:
    def __init__(
        self,
        kgrid,
        ecutwfc, 
        job_proc_desc,
    ):
        self.kgrid:np.ndarray = np.array(kgrid) 
        self.ecutwfc:float = ecutwfc
        self.job_proc_desc: JobProcDesc = job_proc_desc
        
    def get_kgrid(self):
        output = ''
        output += 'K_POINTS automatic\n'
        output += f'{int(self.kgrid[0])} {int(self.kgrid[1])} {int(self.kgrid[2])} 0 0 0\n'
        
        return output 

class DfptInput:
    def __init__(
        self,
        atoms,
        qgrid,
        conv_threshold,
        job_proc_desc,
        njobs=None,
    ):
        self.atoms: AtomsInput = atoms
        self.qgrid: np.array = np.array(qgrid)
        self.conv_threshold: float = conv_threshold
        self.njobs: int = njobs
        self.job_proc_desc: JobProcDesc = job_proc_desc
    
    def get_irr_lim(self, job_idx):
        nmodes: int = self.atoms.atoms.get_number_of_atoms()*3
        
        if self.njobs == None or self.njobs == 1:
            return 1, nmodes
        
        if self.njobs <= nmodes:
            
            modes_per_job = nmodes // self.njobs  
            modes_per_job_last = nmodes - modes_per_job*(self.njobs  - 1)
            
            start_irr = job_idx*modes_per_job + 1 
            last_irr = start_irr + modes_per_job - 1  if job_idx != self.njobs-1 else start_irr + modes_per_job_last - 1 
            
        else:
            raise Exception('njobs <= nmodes should be satisfied.')
        
        return start_irr, last_irr 
    
    def get_nqpts(self):
        with open('kgrid.inp', 'w') as f:
            f.write(f'{self.qgrid[0]:15.10f} {self.qgrid[1]:15.10f} {self.qgrid[2]:15.10f}\n')     
            f.write(f'0.0 0.0 0.0\n')     
            f.write(f'{0.0:15.10f} {0.0:15.10f} {0.0:15.10f}\n')
            f.write(f'{self.atoms.get_scf_cell()}')
            f.write(f'{self.atoms.get_nat()}\n')
            f.write(f'{self.atoms.get_scf_atomic_positions(first_column="atom_index")}')
            f.write(f'0 0 0\n')
            f.write(f'.false.\n')
            f.write(f'.false.\n')
            f.write(f'.false.\n')
            
        os.system('kgrid.x kgrid.inp kgrid.log kgrid.out')
        
        qpts = np.loadtxt('kgrid.log', skiprows=2)
        
        if qpts.ndim == 1 : qpts = qpts.reshape(1, qpts.size)
        
        return qpts.shape[0]        # Return number of qpts. 

class PhbandsInput:
    def __init__(
        self,
        kpath,
        job_proc_desc,
    ):
        self.kpath:np.ndarray = np.array(kpath) 
        self.job_proc_desc: JobProcDesc = job_proc_desc
        
    def get_kpath_str(self):
        output = ''
        output += f'{self.kpath.shape[0]}\n'
        
        for row in self.kpath:
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f}\n'
        
        return output 

class PhDosInput:
    def __init__(
        self,
        qdim,
        job_proc_desc,
    ):
        self.qdim: np.ndarray = np.array(qdim)
        self.job_proc_desc: JobProcDesc = job_proc_desc

class PhModesInput:
    def __init__(
        self,
        qidx,
        job_proc_desc,
    ):
        '''
        qidx: int 
            Starts from 1. It is the index of the irreducibe q-point.  
        '''
        self.qidx: int = qidx 
        self.job_proc_desc: JobProcDesc = job_proc_desc 

class DosInput:
    def __init__(
        self,
        kdim,
        bands,
        job_proc_desc
    ):
        self.kdim: np.ndarray = np.array(kdim)
        self.bands: int = bands
        self.job_proc_desc: JobProcDesc = job_proc_desc

class WannierInput:
    def __init__(
        self,
        atoms,
        kdim,
        kpath_str,
        num_bands,
        num_wann,
        job_proc_desc,
    ):
        self.atoms: AtomsInput = atoms 
        self.kdim: np.ndarray = np.array(kdim)
        self.kpathstr: str = kpath_str
        self.num_bands: int = num_bands
        self.num_wann: int = num_wann
        self.job_proc_desc: JobProcDesc = job_proc_desc
        
    def get_unit_cell_cart(self):
        output = ''
        output += 'begin unit_cell_cart\nAng\n'
        
        for row in self.atoms.atoms.get_cell():
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f}\n'
            
        output += 'end unit_cell_cart\n'
        
        return output 
    
    def get_atoms_frac(self):
        numbers = self.atoms.atoms.get_atomic_numbers()
        scaled_positions = self.atoms.atoms.get_scaled_positions()
        
        output = ''
        output += 'begin atoms_frac\n'
        
        for number, scaled_pos in zip(numbers, scaled_positions):
            output += f'{chemical_symbols[number]} {scaled_pos[0]:15.10f} {scaled_pos[1]:15.10f} {scaled_pos[2]:15.10f}\n'
        
        output += 'end atoms_frac\n'
        
        return output 
        
    def get_mpgrid(self):
        output = f'mp_grid = {int(self.kdim[0])} {int(self.kdim[1])} {int(self.kdim[2])}\n'
        
        return output 
    
    def get_kpoints(self):
        os.system(f'kmesh.pl {self.kdim[0]:15.10f} {self.kdim[1]:15.10f} {self.kdim[2]:15.10f} > kgrid.log')
            
        kgrid = np.loadtxt('kgrid.log', skiprows=2)
        if kgrid.ndim == 1 : kgrid = kgrid.reshape(1, kgrid.size)
        
        output = ''
        output += 'begin kpoints\n'
        
        for row in kgrid:
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f} {row[3]:15.10f}\n'
        
        output += 'end kpoints\n'
        
        return output 
    
    def get_kpoints_qe(self):
        os.system(f'kmesh.pl {self.kdim[0]:15.10f} {self.kdim[1]:15.10f} {self.kdim[2]:15.10f} > kgrid.log')
            
        with open('kgrid.log', 'r') as f: output = f.read()
        
        return output 
    
    def get_kpath(self):
        if not self.kpathstr:
            return ''
        
        special_points = get_special_points(self.atoms.atoms.cell)
        
        output = ''
        output += 'begin kpoint_path\n'
        
        for idx in range(len(self.kpathstr[:-1])):
            char = self.kpathstr[idx]
            point = special_points[char]
            output += f'{char} {point[0]:15.10f} {point[1]:15.10f} {point[2]:15.10f}'
            
            
            char = self.kpathstr[idx+1]
            point = special_points[char]
            output += f'   {char} {point[0]:15.10f} {point[1]:15.10f} {point[2]:15.10f}\n'
        
        output += 'end kpoint_path\n'
        
        return output
    
    def get_bands_plot_flag(self):
        if self.kpathstr:
            return ''
        else:
            return '!'

class DftElbandsInput:
    def __init__(
        self,
        kpath, 
        nbands,
        job_proc_desc,
    ):
        self.kpath:np.ndarray = np.array(kpath) 
        self.nbands: int = nbands 
        self.job_proc_desc: JobProcDesc = job_proc_desc
        
    def get_kgrid_str(self):
        output = ''
        output += 'K_POINTS crystal\n'
        output += f'{self.kpath.shape[0]}\n'
        
        for row in self.kpath:
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f} 1.0\n'
        
        return output 

class WfnGeneralInput:
    def __init__(
        self,
        atoms,
        kdim,
        qshift,
        is_sym,
        bands, 
        job_proc_desc,
        job_proc_desc_pw2bgw=None,
        job_proc_desc_parabands=None,
        parabands_bands=None,
    ):      
        self.atoms:AtomsInput = atoms 
        self.kdim = kdim
        self.qshift = qshift 
        self.is_sym:bool = is_sym
        self.bands = bands  
        self.job_proc_desc: JobProcDesc = job_proc_desc
        self.job_proc_desc_pw2bgw: JobProcDesc = job_proc_desc_pw2bgw
        self.job_proc_desc_parabands: JobProcDesc = job_proc_desc_parabands
        self.parabands_bands = parabands_bands 
        
        self.kgrid:np.ndarray = None 
        self.create_kgrid() 
      
    def create_kgrid(self):
        if self.is_sym==True: # Use kgrid.x
            with open('kgrid.inp', 'w') as f:
                f.write(f'{self.kdim[0]:15.10f} {self.kdim[1]:15.10f} {self.kdim[2]:15.10f}\n')     
                f.write(f'0.0 0.0 0.0\n')     
                f.write(f'{self.qshift[0]:15.10f} {self.qshift[1]:15.10f} {self.qshift[2]:15.10f}\n')
                f.write(f'{self.atoms.get_scf_cell()}')
                f.write(f'{self.atoms.get_nat()}\n')
                f.write(f'{self.atoms.get_scf_atomic_positions(first_column="atom_index")}')
                f.write(f'0 0 0\n')
                f.write(f'.false.\n')
                f.write(f'.false.\n')
                f.write(f'.false.\n')
            
            os.system('kgrid.x kgrid.inp kgrid.log kgrid.out')
            
            self.kgrid = np.loadtxt('kgrid.log', skiprows=2)
            
            if self.kgrid.ndim == 1 : self.kgrid = self.kgrid.reshape(1, self.kgrid.size)
        else: # Use kmesh.pl 
            os.system(f'kmesh.pl {self.kdim[0]:15.10f} {self.kdim[1]:15.10f} {self.kdim[2]:15.10f} > kgrid.log')
            
            self.kgrid = np.loadtxt('kgrid.log', skiprows=2)
            if self.kgrid.ndim == 1 : self.kgrid = self.kgrid.reshape(1, self.kgrid.size)
            
            # Add the qshift to the non-symmetric grid. 
            self.kgrid[:, 0] += self.qshift[0]
            self.kgrid[:, 1] += self.qshift[1]
            self.kgrid[:, 2] += self.qshift[2]
                    
    def get_kgrid_dft_string(self):
        output = ''
        output += 'K_POINTS crystal\n'
        
        num_kpts = self.kgrid.shape[0]
        output += f'{num_kpts}\n'
        
        for row in self.kgrid:
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f} {row[3]:15.10f}\n'
        
        return output 
   
    def get_kgrid_eps_string(self, qshift):
        output = ''
        output += 'begin qpoints\n'
        
        # qshift.
        output += f'{qshift[0]:15.10f} {qshift[1]:15.10f} {qshift[2]:15.10f} 1.0 1\n' 
        
        # qgrid. 
        for row_idx, row in enumerate(self.kgrid):
            if row_idx==0: continue 
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f} 1.0 0\n'
        
        output += 'end\n'
        
        return output 
    
    def get_kgrid_sig_string(self):
        output = ''
        output += 'begin kpoints\n'
        
        # kgrid. 
        for row in self.kgrid:
            output += f'{row[0]:15.10f} {row[1]:15.10f} {row[2]:15.10f} 1.0\n'
        
        output += 'end\n'
        
        return output 
   
class EpwInput:
    def __init__(
        self,
        kgrid_coarse,
        qgrid_coarse,
        kgrid_fine,
        qgrid_fine,
        bands,
        job_proc_desc,
        exec_loc,
        skipped_bands=None,
    ): 
        self.kgrid_coarse:np.ndarray = kgrid_coarse
        self.qgrid_coarse:np.ndarray = qgrid_coarse
        self.kgrid_fine:np.ndarray = kgrid_fine
        self.qgrid_fine:np.ndarray = qgrid_fine
        self.bands = bands  
        self.job_proc_desc: JobProcDesc = job_proc_desc
        self.exec_loc: str = exec_loc
        self.skipped_bands: tuple = skipped_bands
        
class EpsilonInput:
    def __init__(
        self,
        wfn_input,
        bands,
        cutoff,
        wfn_link,
        wfnq_link,
        job_proc_desc,
    ):
        self.wfn_input: WfnGeneralInput = wfn_input
        self.bands = bands  
        self.cutoff = cutoff
        self.wfn_link: str = wfn_link
        self.wfnq_link: str = wfnq_link
        self.job_proc_desc: JobProcDesc = job_proc_desc
        
    def get_qgrid_str(self, qshift):
        return self.wfn_input.get_kgrid_eps_string(qshift=qshift)
        
class SigmaInput:
    def __init__(
        self,
        wfn_input,
        bands,
        band_min,
        band_max,
        cutoff,
        wfn_inner_link,
        job_proc_desc,
    ):
        self.wfn_input: WfnGeneralInput = wfn_input
        self.bands = bands 
        self.band_min = band_min  
        self.band_max = band_max 
        self.cutoff = cutoff
        self.wfn_inner_link: str = wfn_inner_link
        self.job_proc_desc: JobProcDesc = job_proc_desc 
        
    def get_kgrid_str(self):
        return self.wfn_input.get_kgrid_sig_string()

class GwElbandsInput:
    def __init__(
        self,
        val_bands_coarse,
        cond_bands_coarse,
        val_bands_fine,
        cond_bands_fine,
        wfn_co_link,
        wfn_fi_link,
        job_proc_desc,
    ):
        self.val_bands_coarse = val_bands_coarse  
        self.cond_bands_coarse = cond_bands_coarse 
        self.val_bands_fine = val_bands_fine 
        self.cond_bands_fine = cond_bands_fine 
        self.wfn_co_link: str = wfn_co_link
        self.wfn_fi_link: str = wfn_fi_link
        self.job_proc_desc: JobProcDesc = job_proc_desc

class KernelInput:
    def __init__(
        self,
        val_bands_coarse,
        cond_bands_coarse,
        Qshift,
        wfn_co_link,
        job_proc_desc,
    ):
        self.val_bands_coarse = val_bands_coarse  
        self.cond_bands_coarse = cond_bands_coarse 
        self.Qshift: np.ndarray = np.array(Qshift)
        self.wfn_co_link: str = wfn_co_link
        self.job_proc_desc: JobProcDesc = job_proc_desc

class AbsInput:
    def __init__(
        self,
        val_bands_coarse,
        cond_bands_coarse,
        val_bands_fine,
        cond_bands_fine,
        Qshift,
        wfn_co_link,
        wfn_fi_link,
        wfnq_fi_link,
        job_proc_desc,
    ):
        self.val_bands_coarse = val_bands_coarse  
        self.cond_bands_coarse = cond_bands_coarse 
        self.val_bands_fine = val_bands_fine 
        self.cond_bands_fine = cond_bands_fine 
        self.Qshift: np.ndarray = np.array(Qshift)
        self.wfn_co_link: str = wfn_co_link
        self.wfn_fi_link: str = wfn_fi_link
        self.wfnq_fi_link: str = wfnq_fi_link
        self.job_proc_desc: JobProcDesc = job_proc_desc

class PlotxctInput:
    def __init__(
        self,
        hole_position,
        supercell_size,
        state,
        wfn_fi_link,
        wfnq_fi_link,
        job_proc_desc,
    ):
        self.hole_position = hole_position 
        self.supercell_size = supercell_size
        self.state = state 
        self.wfn_fi_link: str = wfn_fi_link
        self.wfnq_fi_link: str = wfnq_fi_link
        self.job_proc_desc: JobProcDesc = job_proc_desc
        
    def get_hole_position_str(self):
        output = f'{self.hole_position[0]:15.10f} {self.hole_position[1]:15.10f} {self.hole_position[2]:15.10f}'
        
        return output 
    
    def get_supercell_size_str(self):
        output = f'{int(self.supercell_size[0])} {int(self.supercell_size[1])} {int(self.supercell_size[2])}'
        
        return output 

class EsfInput:
    def __init__(
        self,
        job_proc_desc,
    ):
        self.job_proc_desc: JobProcDesc = job_proc_desc

class SaveStepInput:
    def __init__(
        self,
        job_proc_desc,
    ):
        self.job_proc_desc: JobProcDesc = job_proc_desc

class EsdInput:
    def __init__(
        self,
        job_proc_desc,
    ):
        self.job_proc_desc: JobProcDesc = job_proc_desc

class Input: 
    def __init__(
        self,
        scheduler,
        atoms: AtomsInput,
        scf: ScfInput,
        dfpt: DfptInput,
        wfn: WfnGeneralInput,
        wfnq: WfnGeneralInput,
        wfnfi: WfnGeneralInput,
        epw: EpwInput,
        wfnqfi: WfnGeneralInput,
        epsilon: EpsilonInput,
        sigma: SigmaInput,
        kernel: KernelInput,
        absorption: AbsInput,
        plotxct: PlotxctInput, 
        phbands: PhbandsInput=None,
        phdos: PhDosInput=None,
        phmodes: PhModesInput=None,
        dos: DosInput=None,
        dftelbands: DftElbandsInput=None,
        wannier: WannierInput=None,
        gwelbands: GwElbandsInput=None,
        esf: EsfInput=None,
        savestep: SaveStepInput=None,
        esd:EsdInput=None,
    ):
        self.scheduler:Scheduler = scheduler,
        self.atoms:AtomsInput = atoms,
        
        self.scf: ScfInput = scf 
        self.dfpt: DfptInput = dfpt 
        self.phbands: PhbandsInput = phbands
        self.phdos: PhDosInput = phdos
        self.phmodes: PhModesInput = phmodes 
        
        self.dos: DosInput = dos 
        self.wannier: WannierInput = wannier
        self.dftelbands: DftElbandsInput = dftelbands 
        
        self.wfn:WfnGeneralInput = wfn
        self.wfnq:WfnGeneralInput = wfnq
        self.wfnfi:WfnGeneralInput = wfnfi
        self.epw:EpwInput = epw
        self.wfnqfi:WfnGeneralInput = wfnqfi
        
        self.epsilon:EpsilonInput = epsilon
        self.sigma:SigmaInput = sigma
        self.gwelbands: GwElbandsInput = gwelbands
        self.kernel:KernelInput = kernel 
        self.absorption:AbsInput = absorption 
        self.plotxct:PlotxctInput = plotxct 
        
        self.esf: EsfInput = esf 
        self.savestep: SaveStepInput = savestep
        self.esd: EsdInput = esd 

class Scf:
    def __init__(self, input: Input):
        self.input: Input = input
    
    def create_files(self): 
        
        write_string_to_file(
            'scf.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='scf'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
!nbnd=10
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
conv_thr=1.0d-8
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.scf.get_kgrid()}
            '''
        ) 
        
        write_string_to_file(
            'job_scf.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.scf.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.scf.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.scf.job_proc_desc)} < scf.in &> scf.in.out  
cp ./tmp/struct.save/data-file-schema.xml ./scf.xml 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf scf.in')
        os.system('rm -rf job_scf.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf scf.in.out')
        os.system('rm -rf scf.xml')
        os.system('rm -rf bandpath.json')
        os.system('rm -rf kgrid.inp kgrid.log kgrid.out')
        os.system('rm -rf *.xsf')

    def run(self):
        run_and_wait_command('./job_scf.sh', self.input)

    def save(self, folder_name): 
        pass 

class Relax:
    def __init__(self, input: Input):
        self.input: Input = input
    
    def create_files(self): 
        
        write_string_to_file(
            'relax.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='vc-relax'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
!nbnd=10
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
cell_dofree='ibrav'
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.scf.get_kgrid()}
            '''
        ) 
        
        write_string_to_file(
            'job_relax.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.scf.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.scf.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.scf.job_proc_desc)} < relax.in &> relax.in.out  
            ''',
        )

    def delete_files(self):
        os.system('rm -rf relax.in')
        os.system('rm -rf job_relax.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf relax.in.out')

    def run(self):
        run_and_wait_command('./job_relax.sh', self.input)

    def save(self, folder_name): 
        pass 

class Md:
    def __init__(self, input: Input):
        self.input: Input = input
    
    def create_files(self): 
        
        write_string_to_file(
            'md.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='md'
tprnfor=.true. 
dt=20 ! ~ 1 fs. au is 50 as. 
nstep=100 ! 100 steps in this 1 fs. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
!nbnd=10
ecutwfc={self.input.scf.ecutwfc}
nosym=.true.
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
pot_extrapolation = 'second-order'
wfc_extrapolation = 'second-order'
ion_temperature   = 'initial'
tempw             = 300
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.scf.get_kgrid()}
            '''
        ) 
        
        write_string_to_file(
            'job_md.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.scf.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.scf.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.scf.job_proc_desc)} < md.in &> md.in.out  
            ''',
        )

    def delete_files(self):
        os.system('rm -rf md.in')
        os.system('rm -rf job_md.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf md.in.out')

    def run(self):
        run_and_wait_command('./job_md.sh', self.input)

    def save(self, folder_name): 
        pass 

class Dfpt:
    def __init__(self, input):
        self.input:Input = input
    
    def write_qpts_create_save(self):
        # Read. 
        with open('./create_save.py', 'r') as f: txt = f.read()
        
        # Replace. 
        nqpt = self.input.dfpt.get_nqpts()
        txt = re.sub(r'nqpt=\d+', f'nqpt={int(nqpt)}', txt)
        
        # Write back. 
        with open('./create_save.py', 'w') as f: txt = f.write(txt)       
    
    def create_njobs(self):
        # Write the start. 
        write_string_to_file(
            f'dfpt_start.in',
            f'''&INPUTPH
outdir='./tmp'
prefix='struct'
ldisp=.true.
nq1={self.input.dfpt.qgrid[0]}
nq2={self.input.dfpt.qgrid[1]}
nq3={self.input.dfpt.qgrid[2]}
tr2_ph={self.input.dfpt.conv_threshold}
fildyn='struct.dyn'
fildvscf='dvscf'
start_irr=0
last_irr=0
/
            '''
        ) 
        
        # Write for njobs. 
        shell_command = ''
        njobs = self.input.dfpt.njobs
        for job_idx in range(njobs):
            start_irr, last_irr = self.input.dfpt.get_irr_lim(job_idx)
            
            write_string_to_file(
            f'dfpt_{job_idx}.in',
            f'''&INPUTPH
outdir='./tmp'
prefix='struct'
ldisp=.true.
nq1={self.input.dfpt.qgrid[0]}
nq2={self.input.dfpt.qgrid[1]}
nq3={self.input.dfpt.qgrid[2]}
tr2_ph={self.input.dfpt.conv_threshold}
fildyn='struct.dyn'
fildvscf='dvscf'
recover=.true.
start_irr={start_irr}
last_irr={last_irr}
/
            '''
        ) 

            total_tasks = self.input.dfpt.job_proc_desc.tasks 
            job_tasks = total_tasks // njobs if job_idx != njobs-1 else total_tasks - (total_tasks // njobs)*(njobs-1)
            split_job_desc = JobProcDesc(
                time=self.input.dfpt.job_proc_desc.time,
                nodes=self.input.dfpt.job_proc_desc.nodes,
                tasks=job_tasks,
                nk=self.input.dfpt.job_proc_desc.nk,
                ni=self.input.dfpt.job_proc_desc.ni,
            )
            
            shell_command += f'{self.input.scheduler.get_parallel_prefix(split_job_desc)}ph.x {self.input.scheduler.get_parallel_infix(split_job_desc)} < dfpt_{job_idx}.in &> dfpt_{job_idx}.in.out &\n'
        
        # Write the last one. 
        write_string_to_file(
            f'dfpt_end.in',
            f'''&INPUTPH
outdir='./tmp'
prefix='struct'
ldisp=.true.
nq1={self.input.dfpt.qgrid[0]}
nq2={self.input.dfpt.qgrid[1]}
nq3={self.input.dfpt.qgrid[2]}
tr2_ph={self.input.dfpt.conv_threshold}
fildyn='struct.dyn'
fildvscf='dvscf'
recover=.true.
/
            '''
        ) 
        
        # Write the job script. 
        write_string_to_file(
            'job_dfpt.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dfpt.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dfpt.job_proc_desc)}ph.x {self.input.scheduler.get_parallel_infix(self.input.dfpt.job_proc_desc)} < dfpt_start.in &> dfpt_start.in.out  
{shell_command}
wait
{self.input.scheduler.get_parallel_prefix(self.input.dfpt.job_proc_desc)}ph.x {self.input.scheduler.get_parallel_infix(self.input.dfpt.job_proc_desc)} < dfpt_end.in &> dfpt_end.in.out  
python3 create_save.py
            ''',
        )
    
    def create_normal_job(self):
        write_string_to_file(
            'dfpt.in',
            f'''&INPUTPH
outdir='./tmp'
prefix='struct'
ldisp=.true.
nq1={self.input.dfpt.qgrid[0]}
nq2={self.input.dfpt.qgrid[1]}
nq3={self.input.dfpt.qgrid[2]}
tr2_ph={self.input.dfpt.conv_threshold}
fildyn='struct.dyn'
fildvscf='dvscf'
/
            '''
        ) 
    
        write_string_to_file(
            'job_dfpt.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dfpt.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dfpt.job_proc_desc)}ph.x {self.input.scheduler.get_parallel_infix(self.input.dfpt.job_proc_desc)} < dfpt.in &> dfpt.in.out  
python3 create_save.py
            ''',
        )
    
    def create_files(self): 
        
        if self.input.dfpt.njobs == None:
            self.create_normal_job()
        else:
            self.create_njobs()

        os.system('cp ./utils/create_save.py ./')
        self.write_qpts_create_save()

    def delete_files(self):
        os.system('rm -rf dfpt_start.in')
        os.system('rm -rf dfpt_start.in.out')
        os.system('rm -rf dfpt_end.in')
        os.system('rm -rf dfpt_end.in.out')
        os.system('rm -rf dfpt.in')
        os.system('rm -rf dfpt.in.out')
        os.system('rm -rf create_save.py')
        os.system('rm -rf job_dfpt.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf ./save')
        os.system('rm -rf struct.dyn*')
        
        
        if self.input.dfpt.njobs != None: 
            for job_idx in range(self.input.dfpt.njobs):
                os.system(f'rm -rf dfpt_{job_idx}.in')
                os.system(f'rm -rf dfpt_{job_idx}.in.out')
            

    def run(self):
        run_and_wait_command('./job_dfpt.sh', self.input)

    def save(self, folder_name): 
        pass 

class PhBands:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        write_string_to_file(
            'q2r.in',
            f'''&INPUT
zasr='crystal'
fildyn='struct.dyn'
flfrc='struct.fc'
/
            '''
        )
        
        write_string_to_file(
            'matdyn_bands.in',
            f'''&INPUT
asr='crystal'
flfrc='struct.fc'
flfrq='struct.freq'
flvec='struct.modes'
q_in_cryst_coord=.true.
/
{self.input.phbands.get_kpath_str()}
            '''
        )
        
        write_string_to_file(
            'job_q2r.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.phbands.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.phbands.job_proc_desc)}q2r.x < q2r.in &> q2r.in.out 
            '''
        )
        
        write_string_to_file(
            'job_matdyn_bands.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.phbands.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.phbands.job_proc_desc)}matdyn.x < matdyn_bands.in &> matdyn_bands.in.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf q2r.in')
        os.system('rm -rf q2r.in.out')
        os.system('rm -rf job_q2r.sh')
        
        os.system('rm -rf matdyn_bands.in')
        os.system('rm -rf matdyn_bands.in.out')
        os.system('rm -rf job_matdyn_bands.sh')
        
        os.system('rm -rf struct.dyn*')
        os.system('rm -rf struct.fc')
        os.system('rm -rf struct.freq')
        os.system('rm -rf struct.modes')
    
    def run(self):
        run_and_wait_command('./job_q2r.sh', self.input)
        run_and_wait_command('./job_matdyn_bands.sh', self.input)
    
    def save(self, folder_name):
        pass 

class PhDos:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        
        write_string_to_file(
            'matdyn_dos.in',
            f'''&INPUT
asr='crystal'
flfrc='struct.fc'
flfrq='struct.dos.freq'
flvec='struct.dos.modes'
dos=.true.
fldos='struct.dos'
nk1={int(self.input.phdos.qdim[0])}
nk2={int(self.input.phdos.qdim[1])}
nk3={int(self.input.phdos.qdim[2])}
/
            '''
        )
        
        write_string_to_file(
            'job_matdyn_dos.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.phdos.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.phdos.job_proc_desc)}matdyn.x < matdyn_dos.in &> matdyn_dos.in.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf matdyn_dos.in')
        os.system('rm -rf matdyn_dos.in.out')
        os.system('rm -rf job_matdyn_dos.sh')
        
        os.system('rm -rf struct.dyn*')
        os.system('rm -rf struct.fc')
        os.system('rm -rf struct.dos.freq')
        os.system('rm -rf struct.dos.modes')
    
    def run(self):
        run_and_wait_command('./job_matdyn_dos.sh', self.input)
    
    def save(self, folder_name):
        pass 

class PhModes:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        
        write_string_to_file(
            'dynmat.in',
            f'''&INPUT
asr='crystal'
fildyn='struct.dyn{self.input.phmodes.qidx}'
filxsf='struct_phmodes.axsf'
/
            '''
        )
        
        write_string_to_file(
            'job_dynmat.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.phmodes.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.phmodes.job_proc_desc)}dynmat.x < dynmat.in &> dynmat.in.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf dynmat.in')
        os.system('rm -rf dynmat.out')
        os.system('rm -rf dynmat.mold')
        os.system('rm -rf input_tmp.in')
        os.system('rm -rf dynmat.in.out')
        os.system('rm -rf job_dynmat.sh')
        
        os.system('rm -rf struct.dyn*')
        os.system('rm -rf struct_phmodes.axsf')
    
    def run(self):
        run_and_wait_command('./job_dynmat.sh', self.input)
    
    def save(self, folder_name):
        pass 

class Dos:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        write_string_to_file(
            'wfndos.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.dos.bands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

K_POINTS automatic 
{int(self.input.dos.kdim[0])} {int(self.input.dos.kdim[1])} {int(self.input.dos.kdim[2])} 0 0 0
            '''
        )
        
        write_string_to_file(
            'job_wfndos.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dos.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dos.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.dos.job_proc_desc)} < wfndos.in &> wfndos.in.out 
            '''
        )
        
        write_string_to_file(
            'dos.in',
            '''&DOS
outdir='./tmp'
prefix='struct'
fildos='struct_dos.dat'
/
            '''
        )
        
        write_string_to_file(
            'job_dos.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dos.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dos.job_proc_desc)}dos.x -pd .true. < dos.in &> dos.in.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf wfndos.in')
        os.system('rm -rf dos.in')
        os.system('rm -rf job_dos.sh')
        os.system('rm -rf job_wfndos.sh')
        
        os.system('rm -rf struct_dos.dat')
        os.system('rm -rf dos.in.out')
        os.system('rm -rf wfndos.in.out')
    
    def run(self):
        run_and_wait_command('./job_wfndos.sh', self.input)
        run_and_wait_command('./job_dos.sh', self.input)
    
    def save(self, folder_name):
        pass 

class Pdos:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        write_string_to_file(
            'pdos.in',
            '''&PROJWFC
outdir='./tmp'
prefix='struct'
filpdos='struct_pdos.dat'
/
            '''
        )
        
        write_string_to_file(
            'job_pdos.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dos.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dos.job_proc_desc)}projwfc.x -pd .true. < pdos.in &> pdos.in.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf pdos.in')
        os.system('rm -rf job_pdos.sh')
        
        os.system('rm -rf struct_pdos.dat*')
        os.system('rm -rf pdos.in.out')
    
    def run(self):
        run_and_wait_command('./job_pdos.sh', self.input)
    
    def save(self, folder_name):
        pass 

class Kpdos:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        write_string_to_file(
            'kpdos.in',
            '''&PROJWFC
outdir='./tmp'
prefix='struct'
kresolveddos=.true.
filpdos='struct_kpdos.dat'
/
            '''
        )
        
        write_string_to_file(
            'job_kpdos.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dos.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dos.job_proc_desc)}projwfc.x -pd .true. < kpdos.in &> kpdos.in.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf kpdos.in')
        os.system('rm -rf job_kpdos.sh')
        
        os.system('rm -rf struct_kpdos.dat*')
        os.system('rm -rf kpdos.in.out')
    
    def run(self):
        run_and_wait_command('./job_kpdos.sh', self.input)
    
    def save(self, folder_name):
        pass  

class Wannier:
    def __init__(self, input: Input):
        self.input = input 
        
    def create_files(self):
        # wfnwannier.
        write_string_to_file(
            'wfnwannier.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.wannier.num_bands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.wannier.get_kpoints_qe()}
            '''
        )
        
        write_string_to_file(
            'job_wfnwannier.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wannier.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wannier.job_proc_desc)}pw.x < wfnwannier.in &> wfnwannier.in.out 
            '''
        )
        
        # wannier pp. 
        write_string_to_file(
            'wannier.win',
            f'''# Structure. 
{self.input.wannier.get_unit_cell_cart()}

{self.input.wannier.get_atoms_frac()}

# kpoints. 
{self.input.wannier.get_mpgrid()}

{self.input.wannier.get_kpoints()}

{self.input.wannier.get_kpath()}
# Bands. 
num_bands = {self.input.wannier.num_bands}
num_wann = {self.input.wannier.num_wann}

# Options. 
auto_projections = .true. 

# Output. 
{self.input.wannier.get_bands_plot_flag()}bands_plot = .true.
wannier_plot = .true. 
write_hr = .true.
write_u_matrices = .true. 
            '''
        )
        
        write_string_to_file(
            'job_wannierpp.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wannier.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wannier.job_proc_desc)}wannier90.x -pp wannier &> wannier.win.pp.out 
            '''
        )
        
        # pw2wannier.        
        write_string_to_file(
            'pw2wannier.in',
            '''&INPUTPP
outdir='./tmp'
prefix='struct'
seedname='wannier'
write_amn=.true.
write_mmn=.true.
write_unk=.true.
scdm_proj=.true.
/
            '''
        )
        
        write_string_to_file(
            'job_pw2wannier.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wannier.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wannier.job_proc_desc)}pw2wannier90.x < pw2wannier.in &> pw2wannier.in.out 
            '''
        )
        
        # wannier. 
        write_string_to_file(
            'job_wannier.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wannier.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wannier.job_proc_desc)}wannier90.x wannier &> wannier.win.out 
            '''
        )
    
    def delete_files(self):
        os.system('rm -rf wannier*')
        os.system('rm -rf UNK*')
        
        os.system('rm -rf pw2wannier.in')
        os.system('rm -rf pw2wannier.in.out')
        os.system('rm -rf wfnwannier.in')
        os.system('rm -rf wfnwannier.in.out')
        
        
        os.system('rm -rf job_wfnwannier.sh')
        os.system('rm -rf job_wannierpp.sh')
        os.system('rm -rf job_pw2wannier.sh')
        os.system('rm -rf job_wannier.sh')
    
    def run(self):
        run_and_wait_command('./job_wfnwannier.sh', self.input)
        run_and_wait_command('./job_wannierpp.sh', self.input)
        run_and_wait_command('./job_pw2wannier.sh', self.input)
        run_and_wait_command('./job_wannier.sh', self.input)
    
    def save(self, folder_name):
        pass 

class DftElbands:
    def __init__(self, input: Input):
        self.input: Input = input
    
    def create_files(self): 
        
        # dft_elbands. 
        write_string_to_file(
            'dft_elbands.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.dftelbands.nbands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.dftelbands.get_kgrid_str()}
            '''
        ) 
        
        write_string_to_file(
            'job_dft_elbands.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dftelbands.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dftelbands.job_proc_desc)}pw.x < dft_elbands.in &> dft_elbands.in.out  
cp ./tmp/struct.xml ./dft_elbands.xml 
            ''',
        )

        # dft_elbands_pw2bgw. 
        write_string_to_file(
            'dft_elbands_pw2bgw.in',
            f'''
&INPUT_PW2BGW
outdir='./tmp'
prefix='struct'
real_or_complex=2
wfng_flag=.true.
wfng_file='WFN_dft_elbands'
wfng_kgrid=.true.
wfng_nk1=0
wfng_nk2=0
wfng_nk3=0
wfng_dk1=0.0
wfng_dk2=0.0
wfng_dk3=0.0
/
            ''',
        )
        
        write_string_to_file(
            'job_dft_elbands_pw2bgw.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.dftelbands.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.dftelbands.job_proc_desc)}pw2bgw.x -pd .true. < dft_elbands_pw2bgw.in &> dft_elbands_pw2bgw.in.out 
cp ./tmp/WFN_dft_elbands ./
wfn2hdf.x BIN WFN_dft_elbands WFN_dft_elbands.h5 
            ''',
        )
    
    def delete_files(self):
        os.system('rm -rf dft_elbands.in')
        os.system('rm -rf job_dft_elbands.sh')
        os.system('rm -rf dft_elbands_pw2bgw.in')
        os.system('rm -rf job_dft_elbands_pw2bgw.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf dft_elbands.in.out')
        os.system('rm -rf dft_elbands_pw2bgw.in.out')
        os.system('rm -rf dft_elbands.xml')
        os.system('rm -rf kgrid.inp kgrid.log kgrid.out')
        os.system('rm -rf WFN_dft_elbands')
        os.system('rm -rf WFN_dft_elbands.h5')
        
        os.system('rm -rf uc_kpath.txt')
        os.system('rm -rf sc_grid.txt')
        os.system('rm -rf sc_Kpath.txt')
        os.system('rm -rf sc_Gshift.txt')

    def run(self):
        run_and_wait_command('./job_dft_elbands.sh', self.input)
        run_and_wait_command('./job_dft_elbands_pw2bgw.sh', self.input)

    def save(self, folder_name): 
        pass 

class Wfn:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        
        # wfn. 
        write_string_to_file(
            'wfn.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.wfn.bands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.wfn.get_kgrid_dft_string()}
            '''
        ) 
        
        write_string_to_file(
            'job_wfn.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfn.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wfn.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.wfn.job_proc_desc)} < wfn.in &> wfn.in.out 
            ''',
        )
        
        # wfn_pw2bgw. 
        write_string_to_file(
            'wfn_pw2bgw.in',
            f'''
&INPUT_PW2BGW
outdir='./tmp'
prefix='struct'
real_or_complex=2
wfng_flag=.true.
wfng_file='WFN'
wfng_kgrid=.true.
wfng_nk1={self.input.wfn.kdim[0]}
wfng_nk2={self.input.wfn.kdim[1]}
wfng_nk3={self.input.wfn.kdim[2]}
wfng_dk1={self.input.wfn.qshift[0]*self.input.wfn.kdim[0]}
wfng_dk2={self.input.wfn.qshift[1]*self.input.wfn.kdim[1]}
wfng_dk3={self.input.wfn.qshift[2]*self.input.wfn.kdim[2]}
rhog_flag=.true.
rhog_file='RHO'
vxcg_flag=.true.
vxcg_file='VXC'
vscg_flag=.true.
vscg_file='VSC'
vkbg_flag=.true.
vkbg_file='VKB'
/
            ''',
        )
        
        write_string_to_file(
            'job_wfn_pw2bgw.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfn.job_proc_desc_pw2bgw)}

{self.input.scheduler.get_parallel_prefix(self.input.wfn.job_proc_desc_pw2bgw)}pw2bgw.x -pd .true. < wfn_pw2bgw.in &> wfn_pw2bgw.in.out 
cp ./tmp/WFN ./
cp ./tmp/RHO ./
cp ./tmp/VXC ./
cp ./tmp/VSC ./
cp ./tmp/VKB ./
            ''',
        )

        # parabands. 
        write_string_to_file(
            'parabands.inp',
            f'''input_wfn_file WFN
output_wfn_file WFN_parabands.h5 

vsc_file VSC 
vkb_file VKB 

number_bands {self.input.wfn.parabands_bands}
            ''',
        )
        
        write_string_to_file(
            'job_parabands.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfn.job_proc_desc_parabands)}

{self.input.scheduler.get_parallel_prefix(self.input.wfn.job_proc_desc_parabands)}parabands.cplx.x &> parabands.inp.out 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf wfn.in')
        os.system('rm -rf job_wfn.sh')
        os.system('rm -rf wfn_pw2bgw.in')
        os.system('rm -rf job_wfn_pw2bgw.sh')
        os.system('rm -rf parabands.inp')
        os.system('rm -rf job_parabands.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf ./WFN')
        os.system('rm -rf ./WFN_parabands.h5')
        os.system('rm -rf ./RHO')
        os.system('rm -rf ./vxc.dat')
        os.system('rm -rf ./VXC')
        os.system('rm -rf ./VSC')
        os.system('rm -rf ./VKB')
        os.system('rm -rf wfn.in.out')
        os.system('rm -rf wfn_pw2bgw.in.out')
        os.system('rm -rf parabands.inp.out')

    def run(self):
        run_and_wait_command('./job_wfn.sh', self.input)
        run_and_wait_command('./job_wfn_pw2bgw.sh', self.input)
        run_and_wait_command('./job_parabands.sh', self.input)

    def save(self, folder_name): 
        pass 

class Wfnq:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        
        # wfnq. 
        write_string_to_file(
            'wfnq.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.wfnq.bands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.wfnq.get_kgrid_dft_string()}
            '''
        ) 
        
        write_string_to_file(
            'job_wfnq.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfnq.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wfnq.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.wfnq.job_proc_desc)} < wfnq.in &> wfnq.in.out 
            ''',
        )
        
        # wfnq_pw2bgw. 
        write_string_to_file(
            'wfnq_pw2bgw.in',
            f'''
&INPUT_PW2BGW
outdir='./tmp'
prefix='struct'
real_or_complex=2
wfng_flag=.true.
wfng_file='WFNq'
wfng_kgrid=.true.
wfng_nk1={self.input.wfnq.kdim[0]}
wfng_nk2={self.input.wfnq.kdim[1]}
wfng_nk3={self.input.wfnq.kdim[2]}
wfng_dk1={self.input.wfnq.qshift[0]*self.input.wfnq.kdim[0]}
wfng_dk2={self.input.wfnq.qshift[1]*self.input.wfnq.kdim[1]}
wfng_dk3={self.input.wfnq.qshift[2]*self.input.wfnq.kdim[2]}
/
            ''',
        )
        
        write_string_to_file(
            'job_wfnq_pw2bgw.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfnq.job_proc_desc_pw2bgw)}

{self.input.scheduler.get_parallel_prefix(self.input.wfnq.job_proc_desc_pw2bgw)}pw2bgw.x -pd .true. < wfnq_pw2bgw.in &> wfnq_pw2bgw.in.out 
cp ./tmp/WFNq ./
wfn2hdf.x BIN WFNq WFNq.h5 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf wfnq.in')
        os.system('rm -rf job_wfnq.sh')
        os.system('rm -rf wfnq_pw2bgw.in')
        os.system('rm -rf job_wfnq_pw2bgw.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf WFNq')
        os.system('rm -rf WFNq.h5')
        os.system('rm -rf wfnq.in.out')
        os.system('rm -rf wfnq_pw2bgw.in.out')

    def run(self):
        run_and_wait_command('./job_wfnq.sh', self.input)
        run_and_wait_command('./job_wfnq_pw2bgw.sh', self.input)

    def save(self, folder_name): 
        pass 

class Wfnfi:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        
        # wfnq. 
        write_string_to_file(
            'wfnfi.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.wfnfi.bands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.wfnfi.get_kgrid_dft_string()}
            '''
        ) 
        
        write_string_to_file(
            'job_wfnfi.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfnfi.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wfnfi.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.wfnfi.job_proc_desc)} < wfnfi.in &> wfnfi.in.out 
            ''',
        )
        
        # wfnq_pw2bgw. 
        write_string_to_file(
            'wfnfi_pw2bgw.in',
            f'''
&INPUT_PW2BGW
outdir='./tmp'
prefix='struct'
real_or_complex=2
wfng_flag=.true.
wfng_file='WFN_fi'
wfng_kgrid=.true.
wfng_nk1={self.input.wfnq.kdim[0]}
wfng_nk2={self.input.wfnq.kdim[1]}
wfng_nk3={self.input.wfnq.kdim[2]}
wfng_dk1={self.input.wfnq.qshift[0]*self.input.wfnq.kdim[0]}
wfng_dk2={self.input.wfnq.qshift[1]*self.input.wfnq.kdim[1]}
wfng_dk3={self.input.wfnq.qshift[2]*self.input.wfnq.kdim[2]}
/
            ''',
        )
        
        write_string_to_file(
            'job_wfnfi_pw2bgw.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfnfi.job_proc_desc_pw2bgw)}

{self.input.scheduler.get_parallel_prefix(self.input.wfnfi.job_proc_desc_pw2bgw)}pw2bgw.x -pd .true. < wfnfi_pw2bgw.in &> wfnfi_pw2bgw.in.out 
cp ./tmp/WFN_fi ./
wfn2hdf.x BIN WFN_fi WFN_fi.h5 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf wfnfi.in')
        os.system('rm -rf job_wfnfi.sh')
        os.system('rm -rf wfnfi_pw2bgw.in')
        os.system('rm -rf job_wfnfi_pw2bgw.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf WFN_fi')
        os.system('rm -rf WFN_fi.h5')
        os.system('rm -rf wfnfi.in.out')
        os.system('rm -rf wfnfi_pw2bgw.in.out')

    def run(self):
        run_and_wait_command('./job_wfnfi.sh', self.input)
        run_and_wait_command('./job_wfnfi_pw2bgw.sh', self.input)

    def save(self, folder_name): 
        pass 

class Epw:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        skipped_bands = self.input.epw.skipped_bands
        skipped_bnds_str = f"bands_skipped='exclude_bands={skipped_bands[0]}:{skipped_bands[1]}'" if skipped_bands != None else ''
        
        num_bands = self.input.epw.bands
            
        
        write_string_to_file(
            'epw.in',
            f'''&INPUTEPW
outdir='./tmp'
prefix='struct'

! kpoints.
nk1={self.input.epw.kgrid_coarse[0]}
nk2={self.input.epw.kgrid_coarse[1]}
nk3={self.input.epw.kgrid_coarse[2]}
nq1={self.input.epw.qgrid_coarse[0]}
nq2={self.input.epw.qgrid_coarse[1]}
nq3={self.input.epw.qgrid_coarse[2]}
nkf1={self.input.epw.kgrid_fine[0]}
nkf2={self.input.epw.kgrid_fine[1]}
nkf3={self.input.epw.kgrid_fine[2]}
nqf1={self.input.epw.qgrid_fine[0]}
nqf2={self.input.epw.qgrid_fine[1]}
nqf3={self.input.epw.qgrid_fine[2]}

! Bands. 
nbndsub={num_bands}
{skipped_bnds_str}

! elph. 
dvscf_dir='./save' 
elph=.true. 
epbwrite=.true. 
epbread=.false.
!prtgkk=.true.

! wannier. 
wannierize=.true. 
proj(1)='random'

! others. 
!temps=300.0
!iverbosity=1
/

            '''
        ) 
        
        write_string_to_file(
            'job_epw.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.epw.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.epw.job_proc_desc)}{self.input.epw.exec_loc} {self.input.scheduler.get_parallel_infix(self.input.epw.job_proc_desc)} < epw.in &> epw.in.out 
cp ./tmp/struct_elph* ./
            ''',
        )

    def delete_files(self):
        os.system('rm -rf epw.in')
        os.system('rm -rf job_epw.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf ./struct*')
        os.system('rm -rf ./decay*')
        os.system('rm -rf ./struct_elph*')
        os.system('rm -rf EPW.bib')
        os.system('rm -rf epwdata.fmt')
        os.system('rm -rf selecq.fmt')
        os.system('rm -rf vmedata.fmt')
        os.system('rm -rf crystal.fmt')
        os.system('rm -rf epw.in.out')

    def run(self):
        run_and_wait_command('./job_epw.sh', self.input)

    def save(self, folder_name): 
        pass 

class Wfnqfi:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        
        # wfnq. 
        write_string_to_file(
            'wfnqfi.in',
            f'''&CONTROL
outdir='./tmp'
prefix='struct'
pseudo_dir='./ONCVPSP/sg15'
calculation='bands'
tprnfor=.true. 
/

&SYSTEM
ibrav=0
ntyp={self.input.atoms.get_ntyp()}
nat={self.input.atoms.get_nat()}
nbnd={self.input.wfnqfi.bands}
ecutwfc={self.input.scf.ecutwfc}
!noncolin=.true.
!lspinorb=.true. 
/

&ELECTRONS
/

&CELL
/

&IONS
/

CELL_PARAMETERS angstrom
{self.input.atoms.get_scf_cell()}

ATOMIC_SPECIES
{self.input.atoms.get_scf_atomic_species()}

ATOMIC_POSITIONS angstrom 
{self.input.atoms.get_scf_atomic_positions()}

{self.input.wfnqfi.get_kgrid_dft_string()}
            '''
        ) 
        
        write_string_to_file(
            'job_wfnqfi.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfnqfi.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.wfnqfi.job_proc_desc)}pw.x {self.input.scheduler.get_parallel_infix(self.input.wfnqfi.job_proc_desc)} < wfnqfi.in &> wfnqfi.in.out 
            ''',
        )
        
        # wfnq_pw2bgw. 
        write_string_to_file(
            'wfnqfi_pw2bgw.in',
            f'''
&INPUT_PW2BGW
outdir='./tmp'
prefix='struct'
real_or_complex=2
wfng_flag=.true.
wfng_file='WFNq_fi'
wfng_kgrid=.true.
wfng_nk1={self.input.wfnqfi.kdim[0]}
wfng_nk2={self.input.wfnqfi.kdim[1]}
wfng_nk3={self.input.wfnqfi.kdim[2]}
wfng_dk1={self.input.wfnqfi.qshift[0]*self.input.wfnqfi.kdim[0]}
wfng_dk2={self.input.wfnqfi.qshift[1]*self.input.wfnqfi.kdim[1]}
wfng_dk3={self.input.wfnqfi.qshift[2]*self.input.wfnqfi.kdim[2]}
/
            ''',
        )
        
        write_string_to_file(
            'job_wfnqfi_pw2bgw.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.wfnqfi.job_proc_desc_pw2bgw)}

{self.input.scheduler.get_parallel_prefix(self.input.wfnqfi.job_proc_desc_pw2bgw)}pw2bgw.x -pd .true. < wfnqfi_pw2bgw.in &> wfnqfi_pw2bgw.in.out 
cp ./tmp/WFNq_fi ./
wfn2hdf.x BIN WFNq_fi WFNq_fi.h5 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf wfnqfi.in')
        os.system('rm -rf job_wfnqfi.sh')
        os.system('rm -rf wfnqfi_pw2bgw.in')
        os.system('rm -rf job_wfnqfi_pw2bgw.sh')
        
        os.system('rm -rf ./tmp')
        os.system('rm -rf WFNq_fi')
        os.system('rm -rf WFNq_fi.h5')
        os.system('rm -rf wfnqfi.in.out')
        os.system('rm -rf wfnqfi_pw2bgw.in.out')

    def run(self):
        run_and_wait_command('./job_wfnqfi.sh', self.input)
        run_and_wait_command('./job_wfnqfi_pw2bgw.sh', self.input)

    def save(self, folder_name): 
        pass 

class Epsilon:
    def __init__(self, input):
        self.input:Input = input

    def create_files(self): 
        write_string_to_file(
            'epsilon.inp',
            f'''# Qpoints 
{self.input.epsilon.get_qgrid_str(self.input.wfnq.qshift)}

# Bands
number_bands {self.input.epsilon.bands}
degeneracy_check_override

# G-Cutoff. 
epsilon_cutoff {self.input.epsilon.cutoff}

# Options

# IO. 
use_wfn_hdf5
            '''
        ) 
        
        write_string_to_file(
            'job_epsilon.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.epsilon.job_proc_desc)}

ln -sf {self.input.epsilon.wfn_link} ./WFN.h5 
ln -sf {self.input.epsilon.wfnq_link} ./WFNq.h5 
{self.input.scheduler.get_parallel_prefix(self.input.epsilon.job_proc_desc)}epsilon.cplx.x &> epsilon.inp.out 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf epsilon.inp')
        os.system('rm -rf job_epsilon.sh')
        
        os.system('rm -rf ./WFN.h5')
        os.system('rm -rf ./WFNq.h5')
        os.system('rm -rf ./epsmat.h5')
        os.system('rm -rf ./eps0mat.h5')
        os.system('rm -rf epsilon.log')
        os.system('rm -rf chi_converge.dat')
        os.system('rm -rf epsilon.inp.out')
        
        os.system('rm -rf checkbz.log')

    def run(self):
        run_and_wait_command('./job_epsilon.sh', self.input)

    def save(self, folder_name): 
        pass 

class Sigma:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        write_string_to_file(
            'sigma.inp',
            f'''# kpoints
{self.input.sigma.get_kgrid_str()}
no_symmetries_q_grid
# use_symmetries_q_grid

# Bands.
number_bands {self.input.sigma.bands}
band_index_min {self.input.sigma.band_min}
band_index_max {self.input.sigma.band_max}
degeneracy_check_override

# G-cutoff
screened_coulomb_cutoff {self.input.sigma.cutoff}


# Options
dont_use_vxcdat

# IO
# verbosity 3
use_wfn_hdf5
            '''
        ) 
        
        write_string_to_file(
            'job_sigma.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.sigma.job_proc_desc)}

ln -sf {self.input.sigma.wfn_inner_link} ./WFN_inner.h5 
{self.input.scheduler.get_parallel_prefix(self.input.sigma.job_proc_desc)}sigma.cplx.x &> sigma.inp.out 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf sigma.inp')
        os.system('rm -rf job_sigma.sh')
        
        os.system('rm -rf ./WFN_inner.h5')
        os.system('rm -rf eqp0.dat')
        os.system('rm -rf eqp1.dat')
        os.system('rm -rf sigma_hp.log')
        os.system('rm -rf ch_converge.dat')
        os.system('rm -rf sigma.inp.out')

    def run(self):
        run_and_wait_command('./job_sigma.sh', self.input)

    def save(self, folder_name): 
        pass 

class Sig2Wan:
    pass 

class GwElbands:
    def __init__(self, input: Input):
        self.input = input 
    
    def create_files(self):
        write_string_to_file(
            'inteqp.inp',
            f'''
number_val_bands_coarse {int(self.input.gwelbands.val_bands_coarse)}
number_cond_bands_coarse {int(self.input.gwelbands.cond_bands_coarse)}
degeneracy_check_override

number_val_bands_fine {int(self.input.gwelbands.val_bands_fine)}
number_cond_bands_fine {int(self.input.gwelbands.cond_bands_fine)}

use_symmetries_coarse_grid
no_symmetries_fine_grid
            '''
        )
    
        write_string_to_file(
            'job_inteqp.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.gwelbands.job_proc_desc)}

ln -sf {self.input.gwelbands.wfn_co_link} ./WFN_co 
ln -sf {self.input.gwelbands.wfn_fi_link} ./WFN_fi 
ln -sf ./eqp1.dat ./eqp_co.dat 
{self.input.scheduler.get_parallel_prefix(self.input.gwelbands.job_proc_desc)}inteqp.cplx.x &> inteqp.inp.out 
mv bandstructure.dat bandstructure_inteqp.dat 
            '''
        )
        
    def delete_files(self):
        os.system('rm -rf inteqp.inp')
        os.system('rm -rf inteqp.inp.out')
        
        os.system('rm -rf WFN_co')
        os.system('rm -rf WFN_fi')
        os.system('rm -rf eqp_co.dat')
        
        os.system('rm -rf bandstructure_inteqp.dat')
        os.system('rm -rf eqp.dat')
        os.system('rm -rf eqp_q.dat')
        os.system('rm -rf dvmat_norm.dat')
        os.system('rm -rf dcmat_norm.dat')
        
        os.system('rm -rf job_inteqp.sh')
    
    def run(self):
        run_and_wait_command('./job_inteqp.sh', input=self.input)
    
    def save(self, folder_name):
        pass 

class Kernel:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        write_string_to_file(
            'kernel.inp',
            f'''# Q-points
exciton_Q_shift 2 {int(self.input.kernel.Qshift[0])} {int(self.input.kernel.Qshift[1])} {int(self.input.kernel.Qshift[2])}
use_symmetries_coarse_grid

# Bands 
number_val_bands {self.input.absorption.val_bands_coarse}
number_cond_bands {self.input.absorption.cond_bands_coarse}
#spinor

# Options
#extended_kernel

# IO. 
use_wfn_hdf5
            '''
        ) 
        
        write_string_to_file(
            'job_kernel.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.kernel.job_proc_desc)}

ln -sf {self.input.kernel.wfn_co_link} WFN_co.h5
{self.input.scheduler.get_parallel_prefix(self.input.kernel.job_proc_desc)}kernel.cplx.x &> kernel.inp.out
            ''',
        )

    def delete_files(self):
        os.system('rm -rf kernel.inp')
        os.system('rm -rf job_kernel.sh')
        
        os.system('rm -rf ./WFN_co.h5')
        os.system('rm -rf bsemat.h5')
        os.system('rm -rf kernel.inp.out')

    def run(self):
        run_and_wait_command('./job_kernel.sh', self.input)

    def save(self, folder_name): 
        pass 

class Absorption:
    def __init__(self, input):
        self.input:Input = input

    def create_files(self): 
        write_string_to_file(
            'absorption.inp',
            f'''# Q-points
exciton_Q_shift 2 {int(self.input.absorption.Qshift[0])} {int(self.input.absorption.Qshift[1])} {int(self.input.absorption.Qshift[2])}
use_symmetries_coarse_grid
no_symmetries_fine_grid
no_symmetries_shifted_grid

# Bands
number_val_bands_coarse {self.input.absorption.val_bands_coarse}
number_cond_bands_coarse {self.input.absorption.cond_bands_coarse}
number_val_bands_fine {self.input.absorption.val_bands_fine}
number_cond_bands_fine {self.input.absorption.cond_bands_fine}
degeneracy_check_override
#spinor

# Options
diagonalization
#use_velocity
use_momentum
polarization {self.input.wfnq.qshift[0]} {self.input.wfnq.qshift[1]} {self.input.wfnq.qshift[2]}
eqp_co_corrections
dump_bse_hamiltonian

# IO
use_wfn_hdf5

# Output
energy_resolution 0.01
write_eigenvectors 10
            '''
        ) 
        
        write_string_to_file(
            'job_absorption.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.absorption.job_proc_desc)}

ln -sf {self.input.absorption.wfn_co_link} WFN_co.h5 
ln -sf {self.input.absorption.wfn_fi_link} WFN_fi.h5 
ln -sf {self.input.absorption.wfnq_fi_link} WFNq_fi.h5 
ln -sf eqp1.dat eqp_co.dat 
{self.input.scheduler.get_parallel_prefix(self.input.absorption.job_proc_desc)}absorption.cplx.x &> absorption.inp.out 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf absorption.inp')
        os.system('rm -rf job_absorption.sh')
        
        os.system('rm -rf ./WFN_fi.h5')
        os.system('rm -rf ./WFNq_fi.h5')
        os.system('rm -rf eigenvalues.dat')
        os.system('rm -rf eigenvalues_noeh.dat')
        os.system('rm -rf absorption_eh.dat')
        os.system('rm -rf absorption_noeh.dat')
        os.system('rm -rf dvmat_norm.dat')
        os.system('rm -rf dcmat_norm.dat')
        os.system('rm -rf eqp_co.dat')
        os.system('rm -rf eqp.dat')
        os.system('rm -rf eqp_q.dat')
        os.system('rm -rf bandstructure.dat')
        os.system('rm -rf eigenvectors.h5')
        os.system('rm -rf hbse*.h5')
        os.system('rm -rf x.dat')
        os.system('rm -rf epsdiag.dat')
        os.system('rm -rf dtmat')
        os.system('rm -rf vmtxel')
        os.system('rm -rf absorption.inp.out')

    def run(self):
        run_and_wait_command('./job_absorption.sh', self.input)

    def save(self, folder_name): 
        pass 

class PlotXct:
    def __init__(self, input):
        self.input:Input = input
        
    def create_files(self): 
        write_string_to_file(
            'plotxct.inp',
            f'''# Cell parameters.
hole_position {self.input.plotxct.get_hole_position_str()}
supercell_size {self.input.plotxct.get_supercell_size_str()}

# Q-points. 
# q_shift
no_symmetries_fine_grid
no_symmetries_shifted_grid

# Bands and state. 
plot_spin 1
plot_state {self.input.plotxct.state}
#spinor
#electron_spin 1
#hole_spin 2

# Output. 

# IO
use_wfn_hdf5
            '''
        ) 
        
        write_string_to_file(
            'job_plotxct.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.plotxct.job_proc_desc)}


ln -sf {self.input.plotxct.wfn_fi_link} WFN_fi.h5 
ln -sf {self.input.plotxct.wfnq_fi_link} WFNq_fi.h5 
{self.input.scheduler.get_parallel_prefix(self.input.plotxct.job_proc_desc)}plotxct.cplx.x &> plotxct.inp.out 
volume.py ./scf.in espresso *.a3Dr a3dr plotxct.xsf xsf false abs2 true 
            ''',
        )

    def delete_files(self):
        os.system('rm -rf plotxct.inp')
        os.system('rm -rf job_plotxct.sh')
        
        os.system('rm -rf *.a3Dr')
        os.system('rm -rf plotxct.xsf')
        os.system('rm -rf plotxct.inp.out')

    def run(self):
        run_and_wait_command('./job_plotxct.sh', self.input)

    def save(self, folder_name): 
        pass 

class AbsorptionQ:
    pass 

class XctPh:
    pass 

class Esf:
    def __init__(self, input):
        self.input:Input = input

    def create_files(self): 
            
        write_string_to_file(
            'job_esf.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.esf.job_proc_desc)}

{self.input.scheduler.get_parallel_prefix(self.input.esf.job_proc_desc)}python3 calc_esf.py &> esf.out 
            ''',
        )
        
        os.system('cp ./utils/calc_esf.py ./')

    def delete_files(self):
        os.system('rm -rf calc_esf.py')
        os.system('rm -rf job_esf.sh')
        
        os.system('rm -rf esf.h5')
        os.system('rm -rf esf.out')

    def run(self):
        run_and_wait_command('./job_esf.sh', self.input)

    def save(self, folder_name): 
        pass 

class SaveStep:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        
        write_string_to_file(
            'job_savestep.sh',
            f'''#!/bin/bash
{self.input.scheduler.get_job_header(self.input.savestep.job_proc_desc)}

folder_name="1"
{self.input.scheduler.get_parallel_prefix(self.input.savestep.job_proc_desc)}python3 workflow.py --save $folder_name &> savestep.out  
            ''',
        )

    def delete_files(self):
        os.system('rm -rf job_savestep.sh')
        
        os.system('rm -rf savestep.out')

    def run(self):
        run_and_wait_command('./job_savestep.sh', self.input)

    def save(self, folder_name): 
        pass 

class Esd:
    def __init__(self, input):
        self.input:Input = input
    
    def create_files(self): 
        
        write_string_to_file(
            'job_esd.sh',
            f'''#!/bin/bash
python3 calc_esd.py &> esd.out  
            ''',
        )
        
        os.system('cp ./utils/calc_esd.py ./')

    def delete_files(self):
        os.system('rm -rf calc_esd.py')
        os.system('rm -rf job_esd.sh')
        os.system('rm -rf job_all_esd.out')
        
        os.system('rm -rf esd.out')
        os.system('rm -rf esd_log.txt')
        os.system('rm -rf esd.traj')
        os.system('rm -rf ./esd_save')

    def run(self):
        run_and_wait_command('./job_esd.sh', self.input)

    def save(self, folder_name): 
        pass 

#endregion

#region: Main.
if __name__ == "__main__":
    main()
#endregion