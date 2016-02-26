tests: tests.hs Mdiscr.hs Mdiscr4.hs Helpers.hs
	ghc -odir tmp -hidir tmp $^

testsSegm: testsSegm.hs MdiscrSegmented.hs Helpers.hs
	ghc -odir tmp -hidir tmp $^

clean:
	rm -f tests testsSegm tmp/*
