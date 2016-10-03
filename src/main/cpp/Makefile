CC=g++
CFLAGS=-g -Wall -shared -fPIC

# Assumes xgboost and rabit libraries are installed
# directly to /usr/local/lib or another path specified
# by /etc/ld.so.conf.d/something*
# Similarly for the include directory

LIBS=-lxgboost

SRC=xgboost_sharp
TARGET=libxgboost_sharp

all: $(TARGET).so

$(TARGET).so: $(SRC).cpp
	$(CC) $(CFLAGS) -o $(TARGET).so $(SRC).cpp $(LIBS)

clean:
	$(RM) $(TARGET).so
