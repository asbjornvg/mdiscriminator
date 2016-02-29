MdiscrTest: MdiscrTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

CoalescedTest: CoalescedTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

clean:
	rm -f MdiscrTest CoalescedTest


# For synchronizing between my computer and the gpu01 machine.

get:
	rsync -avu --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

get_dry:
	rsync -navu --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

put_delete:
	rsync -avu --delete --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr

put_delete_dry:
	rsync -navu --delete --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr
