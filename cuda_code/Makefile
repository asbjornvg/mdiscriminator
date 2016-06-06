%Thrust: %.cu
	nvcc -O3 -DNDEBUG -arch=sm_20 -DTHRUST -o $@ $<

KERNEL_CONSTANTS = -DMAX_CHUNK=384 -DMAP_X=64 -DMAP_Y=1 -DWRITE_X=32 -DWRITE_Y=32 -DCOMPACT_REPRESENTATION

MdiscrTest: MdiscrTest.cu
	nvcc -O3 -DNDEBUG -arch=sm_20 -o $@ $<

MdiscrTest2: MdiscrTest2.cu
	nvcc -O3 -DNDEBUG $(KERNEL_CONSTANTS) -arch=sm_20 -o $@ $<

%: %.cu
	nvcc -O3 -DNDEBUG -arch=sm_20 -o $@ $<
	#nvcc -O3 -arch=sm_20 -o $@ $<
	#nvcc -g -G -arch=sm_20 -o $@ $<

OpenmpMdiscrTest: OpenmpMdiscrTest.cpp
	g++ -fopenmp -O3 -DNDEBUG -o $@ $<

%: %.cpp
	g++ -O3 -DNDEBUG -o $@ $<

# For debugging, use "nvcc -g -G -arch=sm_20 -o $@ $<"

clean:
	rm -f MdiscrTest_baseline MdiscrTest_optimized MdiscrTest_optimized_clean MdiscrTest_tuples MdiscrTest_arrays MdiscrTestThrust MdiscrSegmTest MdiscrSegmTestThrust SeqMdiscrTest CoalescedTest ThrustTest OpenmpMdiscrTest parbb


# For synchronizing between my computer and the gpu0* machines.

RSYNC_FLAGS    = -avu
RSYNC_DRY      = -n
RSYNC_DELETE   = --delete
RSYNC_EXCLUDES = --exclude 'a.out' --exclude '*~'
GPU            = gpu02
GPU_DIR        = $(GPU):mdiscr
CURRENT_DIR    = .

get:
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(GPU_DIR)/ $(CURRENT_DIR)

get_dry:
	rsync $(RSYNC_DRY) $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(GPU_DIR)/ $(CURRENT_DIR)

put:
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(CURRENT_DIR)/ $(GPU_DIR)

put_dry:
	rsync $(RSYNC_DRY) $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(CURRENT_DIR)/ $(GPU_DIR)

put_delete:
	rsync $(RSYNC_FLAGS) $(RSYNC_DELETE) $(RSYNC_EXCLUDES) $(CURRENT_DIR)/ $(GPU_DIR)

put_delete_dry:
	rsync $(RSYNC_DRY) $(RSYNC_FLAGS) $(RSYNC_DELETE) $(RSYNC_EXCLUDES) $(CURRENT_DIR)/ $(GPU_DIR)
