clean:
	rm -rf build

build:
	cmake -S src/ -B build/
