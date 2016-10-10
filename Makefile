all:
	mkdir -p lib
	mkdir -p bin
	cd src/main/cpp && make all
	cd src/main/cs && make all

clean:
	rm -f bin/*
	rm -f lib/*
	cd src/main/cpp && make clean
	cd src/main/cs && make clean

