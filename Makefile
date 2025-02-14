all:
	g++ -I ${BOOST_ROOT} main.cpp -o target/main

matmul:
	g++ -I ${BOOST_ROOT} matmul.cpp -o target/matmul

clean:
	rm -fr target/*