%Thrust: %.cu
	nvcc -O3 -arch=sm_20 -DTHRUST -o $@ $<

%: %.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

OpenmpMdiscrTest: OpenmpMdiscrTest.cpp
	g++ -fopenmp -O3 -o $@ $<

%: %.cpp
	g++ -O3 -o $@ $<

# For debugging, use "nvcc -g -G -arch=sm_20 -o $@ $<"

clean:
	rm -f MdiscrTest MdiscrTestThrust MdiscrSegmTest MdiscrSegmTestThrust SeqMdiscrTest CoalescedTest ThrustTest ThrustTestDebug


# For synchronizing between my computer and the gpu01 machine.

RSYNC_FLAGS    = -avu
RSYNC_DRY      = -n
RSYNC_DELETE   = --delete
RSYNC_EXCLUDES = --exclude 'a.out' --exclude '*~'
GPU01_DIR      = gpu01:mdiscr
CURRENT_DIR    = .

get:
	rsync $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(GPU01_DIR)/ $(CURRENT_DIR)

get_dry:
	rsync $(RSYNC_DRY) $(RSYNC_FLAGS) $(RSYNC_EXCLUDES) $(GPU01_DIR)/ $(CURRENT_DIR)

put_delete:
	rsync $(RSYNC_FLAGS) $(RSYNC_DELETE) $(RSYNC_EXCLUDES) $(CURRENT_DIR)/ $(GPU01_DIR)

put_delete_dry:
	rsync $(RSYNC_DRY) $(RSYNC_FLAGS) $(RSYNC_DELETE) $(RSYNC_EXCLUDES) $(CURRENT_DIR)/ $(GPU01_DIR)
