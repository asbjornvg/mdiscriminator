#!/usr/bin/env python

from __future__ import division
import shlex, sys, os, subprocess, math, re

num_elems = 50000000
sample_runs = 5
cuda_filename = "MdiscrTest_optimized_clean.cu"
# The name of the binary is simply everything before the dot.
binary_filename = re.search(r"^([^.]+)", cuda_filename).group(1)

def compile(*args):
    cmd = shlex.split("nvcc -O3 -DNDEBUG -DNUM_CLASSES={} -DPACKED_{} -DMAX_CHUNK={} -DMAP_X={} -DMAP_Y={} -DWRITE_X={} -DWRITE_Y={} -arch=sm_20 -o {} {}".format(*(args+(binary_filename,cuda_filename))))
    print cmd
    subprocess.check_call(cmd)

def run_samples(*args):
    # total = 0
    results = []
    compile(*args)
    cmd = shlex.split("./{} {}".format(binary_filename,num_elems))
    print cmd
    with open(os.devnull, 'w') as DEVNULL:
        for _ in range(sample_runs):
            try:
                # r = subprocess.check_output(cmd, stderr=DEVNULL)
                # output = int(r)
                # results.append(output)
                # total += output
                r = subprocess.check_output(cmd, stderr=DEVNULL)
                output = [int(x) for x in r.split()]
                results.append(output)
            except subprocess.CalledProcessError:
                output = None
                break
    if output is None:
        s = "# Failed run on {} elements [with MAX_CHUNK = {}, MAP_X = {}, MAP_Y = {}, WRITE_X = {}, WRITE_Y = {}]".format(num_elems, *args[2:])
        print s
        return s
    else:
        # mean = total / sample_runs
        # sd = 0.0
        # for res in results:
        #     sd += (res - mean) ** 2
        # sd = math.sqrt( sd / (sample_runs-1) )
        # percent = sd/mean*100
        
        # Sort and remove outliers.
        results = sorted(results, key=lambda r: r[0])[1:-1]
        
        totals = [0, 0, 0]
        for res in results:
            totals = [x + y for (x, y) in zip(totals, res)]
        
        means = [total / len(results) for total in totals]
        sd = 0.0 # SD for total output only
        for res in results:
            sd += (res[0] - means[0]) ** 2
        sd = math.sqrt( sd / (len(results)-1) )
        percent = sd/means[0]*100
        
        (max_chunk, map_x, map_y, write_x, write_y) = args[2:]
        s = "{:.2f} {:.2f} {:.2f} {} {} {} {} # Total mean, map mean, write mean, MAX_CHUNK, MAP_X, MAP_Y, WRITE_Y of {} runs on {} elements [with WRITE_X = {}], SD = {:.3f}% for totals".format(round(means[0],2), round(means[1],2), round(means[2],2), max_chunk, map_x, map_y, write_y, sample_runs, num_elems, write_x, round(percent,3))
        return s

def main(argv):
    for (num_classes,version,fldr) in [(2, "V2", "uint16_t_arr/MyInt_2"),
                                       (4, "V2", "uint16_t_arr/MyInt_4"),
                                       (6, "V2", "uint16_t_arr/MyInt_6"),
                                       (8, "V2", "uint16_t_arr/MyInt_8"),
                                       (10, "V2", "uint16_t_arr/MyInt_10"),
                                       (12, "V2", "uint16_t_arr/MyInt_12"),
                                       (14, "V2", "uint16_t_arr/MyInt_14"),
                                       (16, "V2", "uint16_t_arr/MyInt_16"),
                                       (2, "V3", "uint64_t_arr/MyInt_2"),
                                       (4, "V3", "uint64_t_arr/MyInt_4"),
                                       (6, "V3", "uint64_t_arr/MyInt_6"),
                                       (8, "V3", "uint64_t_arr/MyInt_8"),
                                       (10, "V3", "uint64_t_arr/MyInt_10"),
                                       (12, "V3", "uint64_t_arr/MyInt_12"),
                                       (14, "V3", "uint64_t_arr/MyInt_14"),
                                       (16, "V3", "uint64_t_arr/MyInt_16")]:
        folder = "output/timings_compact/after_race_bug/{}/".format(fldr)
        print "Creating folder '{}'".format(folder)
        os.makedirs(folder, 0775)
        cmd = shlex.split("cp run.py {}".format(folder))
        print cmd
        subprocess.check_call(cmd)
        for max_chunk in [96, 192, 384]:
            filename = folder + "output{}.out".format(max_chunk)
            with open(filename, 'a+', 0) as f:
                for map_x in range(64, 257, 64):
                    for map_y in [2**p for p in range(0,4,1) if map_x * (2**p) <= 1024]:
                        for write_y in [4, 8, 16, 32]:
                            s = run_samples(num_classes, version, max_chunk, map_x, map_y, 32, write_y)
                            f.write("{}\n".format(s))

if __name__ == "__main__":
   main(sys.argv[1:])
