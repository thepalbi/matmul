all: matmul

matmul:
	g++ -I ${BOOST_ROOT} matmul.cpp -o target/matmul

clean:
	rm -fr target/*