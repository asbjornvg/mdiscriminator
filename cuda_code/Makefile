MdiscrTest: MdiscrTest.cu
	nvcc -O3 -arch=sm_20 -o $@ $<

clean:
	rm MdiscrTest

get:
	rsync -avu --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

get_dry:
	rsync -navu --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

# put:
# 	rsync -avu --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr

# put_dry:
# 	rsync -navu --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr

# sync: get put

# get_delete:
# 	rsync -avu --delete --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

# get_delete_dry:
# 	rsync -navu --delete --exclude 'a.out' --exclude '*~' gpu01:mdiscr/ .

put_delete:
	rsync -avu --delete --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr

# sync_delete: get_delete put_delete

put_delete_dry:
	rsync -navu --delete --exclude 'a.out' --exclude '*~' ./ gpu01:mdiscr
