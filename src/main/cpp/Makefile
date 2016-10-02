CC=g++
CFLAGS=-g -Wall

# Assumes xgboost and rabit libraries are installed
# directly to /usr/local/lib or another path specified
# by /etc/ld.so.conf.d/something*
# Similarly for the include directory

LIBS=-lxgboost

TARGET=xgboost_sharp

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET)  $(TARGET).cpp $(LIBS)

clean:
	$(RM) $(TARGET)
