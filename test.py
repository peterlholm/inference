"########## testing ############"
from pathlib import Path
from shutil import rmtree, copytree
import time

from inference.process_input import process


_PERF = True

def process_testimage():
    "process i image folder"
    testfolder = Path(__file__).parent / 'testdata/testtarget/render0'
    tmp_folder = Path(__file__).parent / 'tmp'
    rmtree(tmp_folder, ignore_errors=True)
    copytree(testfolder, tmp_folder)
    myoutfolder = tmp_folder / 'out'
    myoutfolder.mkdir()
    if _PERF:
        proc_st = time.process_time()
        st_time = time.time()
    process(tmp_folder, myoutfolder)
    if _PERF:
        end_time = time.time()
        proc_end = time.process_time()
        print("CPU exec time:", proc_end-proc_st, "seconds")
        print("Elapsed time:", end_time-st_time, "seconds")
    print("Data processed")

def process_image_set():
    "process a image folder set"
    tmp_folder = Path(__file__).parent / 'tmp'
    rmtree(tmp_folder, ignore_errors=True)
    tmp_folder.mkdir()
    testset_folder = Path(__file__).parent / 'testdata/testtarget'
    folders = sorted(testset_folder.glob('*'))
    if _PERF:
        proc_st = time.process_time()
        st_time = time.time()
    for fld in folders:
        print(fld)
        copytree(fld, tmp_folder / fld.name)
        # myoutfolder = tmp_folder / 'out'
        # myoutfolder.mkdir()
        process(tmp_folder / fld.name)
    if _PERF:
        end_time = time.time()
        proc_end = time.process_time()
        print("CPU exec time:", proc_end-proc_st, "seconds")
        print("Elapsed time:", end_time-st_time, "seconds")
    print("Data processed")

if __name__=='__main__':
    #process_testimage()
    process_image_set()
