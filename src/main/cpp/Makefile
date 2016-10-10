CC=g++
CFLAGS=-g -Wall -shared -fPIC

# Assumes xgboost and rabit libraries are installed
# directly to /usr/local/lib or another path specified
# by /etc/ld.so.conf.d/something*
# Similarly for the include directory

LIBDIR=../../../lib
LIBS=-lxgboost

SRC=xgboost_sharp
TARGET=libxgboost_sharp

all: $(LIBDIR)/$(TARGET).so

$(LIBDIR)/$(TARGET).so: $(SRC).cpp
	$(CC) $(CFLAGS) -o $(LIBDIR)/$(TARGET).so $(SRC).cpp $(LIBS)

clean:
	$(RM) $(LIBDIR)/$(TARGET).so
