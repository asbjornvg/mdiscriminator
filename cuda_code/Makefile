MdiscrTest: MdiscrTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

MdiscrTestThrust: MdiscrTest.cu
	nvcc -O3 -arch=sm_20 -DTHRUST -o $@ $<

MdiscrSegmTest: MdiscrSegmTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

MdiscrSegmTestThrust: MdiscrSegmTest.cu
	nvcc -O3 -arch=sm_20 -DTHRUST -o $@ $<

SeqMdiscrTest: SeqMdiscrTest.cpp
	g++ -O3 -o $@ $<

OpenmpMdiscrTest: OpenmpMdiscrTest.cpp
	g++ -fopenmp -O3 -o $@ $<

CoalescedTest: CoalescedTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

ThrustTest: ThrustTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

ThrustTestDebug: ThrustTest.cu
	nvcc -g -G -arch=sm_20 -o $@ $<

clean:
	rm -f MdiscrTest MdiscrTestThrust MdiscrSegmTest MdiscrSegmTestThrust SeqMdiscrTest CoalescedTest ThrustTest ThrustTestDebug


# For synchronizing between my computer and the gpu01 machine.

get:
	rsync -avu --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

get_dry:
	rsync -navu --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

put_delete:
	rsync -avu --delete --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr

put_delete_dry:
	rsync -navu --delete --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr
